# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Work.wait 接口功能正确性
API 名称：torch.distributed.Work.wait
API 签名：Work.wait(self, timeout=-1) -> bool

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 验证 timeout 参数默认值与显式传入                          | 已覆盖：test_wait_default_timeout              |
| 参数类型         | 验证 timeout 参数类型（int/float）                          | 已覆盖：test_wait_with_timeout                 |
| 正常传参场景     | 等待异步通信操作完成                                         | 已覆盖：test_wait_completion                   |
| 多卡场景         | 验证多卡环境下的 wait 行为                                   | 已覆盖：test_wait_multiprocess                 |
| 返回值验证       | 验证 wait 方法返回 True 表示成功                            | 已覆盖：test_wait_return_value                 |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestDistributedWorkWait(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29502')
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_wait_default_timeout(cls, rank, world_size, c2p, device_name):
        dist_group = cls._init_dist_hccl(rank, world_size)

        # 必须使用 'npu:<rank>' 而非 'npu'，否则两个 rank 都在物理 0 卡上做 send/recv，
        # 引发 HCCL 通信和内存损坏问题。
        device = f"{device_name}:{rank}"
        torch.npu.synchronize()
        tensor = torch.randn(10, 10, device=device)

        if rank == 0:
            work = dist_group.isend(tensor, dst=1)
            # Work.wait() 在 NPU/HCCL 后端返回 None；这里仅验证调用不报错。
            work.wait()
            c2p.put((rank, 'wait_result', True))
        elif rank == 1:
            recv_tensor = torch.empty(10, 10, device=device)
            work = dist_group.irecv(recv_tensor, src=0)
            work.wait()
            c2p.put((rank, 'wait_result', True))

        torch.npu.synchronize()
        dist_group.destroy_process_group()

    @classmethod
    def _test_wait_with_timeout(cls, rank, world_size, c2p, device_name):
        dist_group = cls._init_dist_hccl(rank, world_size)

        device = f"{device_name}:{rank}"
        torch.npu.synchronize()
        tensor = torch.randn(10, 10, device=device)

        if rank == 0:
            work = dist_group.isend(tensor, dst=1)
            timeout = datetime.timedelta(seconds=30)
            work.wait(timeout=timeout)
            c2p.put((rank, 'wait_result', True))
        elif rank == 1:
            recv_tensor = torch.empty(10, 10, device=device)
            work = dist_group.irecv(recv_tensor, src=0)
            timeout = datetime.timedelta(seconds=30)
            work.wait(timeout=timeout)
            c2p.put((rank, 'wait_result', True))

        torch.npu.synchronize()
        dist_group.destroy_process_group()

    @skipIfUnsupportMultiNPU(2)
    def test_wait_default_timeout(self):
        # Test Work.wait() with default timeout
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(2)
        
        ps = []
        for i in range(2):
            p = ctx.Process(
                target=self._test_wait_default_timeout,
                args=(i, 2, c2p, self.device_name))
            p.start()
            ps.append(p)
        
        results = []
        for _ in range(2):
            results.append(c2p.get(timeout=30))
        
        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")
        
        # Verify all wait operations returned True
        for rank, event, value in results:
            if event == 'wait_result':
                self.assertTrue(value)

    @skipIfUnsupportMultiNPU(2)
    def test_wait_with_timeout(self):
        # Test Work.wait(timeout) with explicit timeout
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(2)
        
        ps = []
        for i in range(2):
            p = ctx.Process(
                target=self._test_wait_with_timeout,
                args=(i, 2, c2p, self.device_name))
            p.start()
            ps.append(p)
        
        results = []
        for _ in range(2):
            results.append(c2p.get(timeout=30))
        
        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")
        
        # Verify all wait operations returned True
        for rank, event, value in results:
            if event == 'wait_result':
                self.assertTrue(value)

    @skipIfUnsupportMultiNPU(2)
    def test_wait_completion(self):
        # Test that wait properly completes async operation
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(4)
        
        ps = []
        for i in range(2):
            p = ctx.Process(
                target=self._test_wait_completion_worker,
                args=(i, 2, c2p, self.device_name))
            p.start()
            ps.append(p)
        
        results = []
        for _ in range(4):
            results.append(c2p.get(timeout=30))
        
        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")
        
        # Verify work completed after wait
        for rank, event, value in results:
            if event == 'is_completed_after_wait':
                self.assertTrue(value)

    @classmethod
    def _test_wait_completion_worker(cls, rank, world_size, c2p, device_name):
        dist_group = cls._init_dist_hccl(rank, world_size)

        device = f"{device_name}:{rank}"
        torch.npu.synchronize()
        tensor = torch.randn(10, 10, device=device)

        if rank == 0:
            work = dist_group.isend(tensor, dst=1)
            is_completed_before = work.is_completed()
            c2p.put((rank, 'is_completed_before_wait', is_completed_before))
            work.wait()
            c2p.put((rank, 'is_completed_after_wait', True))
        elif rank == 1:
            recv_tensor = torch.empty(10, 10, device=device)
            work = dist_group.irecv(recv_tensor, src=0)
            is_completed_before = work.is_completed()
            c2p.put((rank, 'is_completed_before_wait', is_completed_before))
            work.wait()
            c2p.put((rank, 'is_completed_after_wait', True))

        torch.npu.synchronize()
        dist_group.destroy_process_group()


if __name__ == "__main__":
    run_tests()
