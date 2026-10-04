"""并行读取测试：核算移出读事务后，长核算不独占数据库访问。

- 审计在“读视图外”完成几何核算：核算期间写操作不排队；
- 两个审计请求可真正并行（各用独立读连接）；
- 读视图建立/查询失败后连接归还，反复失败不会耗尽连接池。
"""
import threading

import pytest

from tests.conftest import DAY_WINDOWS, SQUARE


def _seed(client):
    r = client.post(
        "/api/v1/zones",
        json={"zone_id": "par", "exterior": SQUARE, "holes": [], "windows": DAY_WINDOWS},
    )
    assert r.status_code == 201
    for tid in ("par-a", "par-b"):
        r = client.post(
            f"/api/v1/trips/{tid}/points",
            json={
                "points": [
                    {"point_id": "p1", "ts": "2026-09-30T00:00:00Z", "x": 10, "y": 50},
                    {"point_id": "p2", "ts": "2026-09-30T00:02:00Z", "x": 90, "y": 50},
                ]
            },
        )
        assert r.status_code == 200, r.text


def _co_occupancy_body():
    return {
        "trip_ids": ["par-a", "par-b"],
        "since": "2026-09-30T00:00:00Z",
        "until": "2026-09-30T00:05:00Z",
        "threshold": 2,
    }


def test_long_audit_computation_does_not_block_writes(client, monkeypatch):
    """审计核算在读事务外进行：核算期间区域登记/轨迹写入不排队。"""
    from app.main import service

    _seed(client)

    entered = threading.Event()
    release = threading.Event()

    original = service._coverage_intervals

    def gated(zrow, points, lo, hi):
        if getattr(gated, "_armed", False):
            entered.set()
            assert release.wait(timeout=10), "核算门未释放"
        return original(zrow, points, lo, hi)

    monkeypatch.setattr(service, "_coverage_intervals", gated)

    write_done = threading.Event()
    write_result = {}

    def audit():
        gated._armed = True
        r = client.post("/api/v1/zones/par/co-occupancy", json=_co_occupancy_body())
        write_result["audit_status"] = r.status_code

    def writer():
        # 等待审计已离开读视图、进入事务外几何核算
        assert entered.wait(timeout=10), "审计未进入核算阶段"
        r = client.post(
            "/api/v1/zones",
            json={
                "zone_id": "par-other",
                "exterior": SQUARE,
                "holes": [],
                "windows": DAY_WINDOWS,
            },
        )
        write_result["write_status"] = r.status_code
        write_done.set()

    threads = [threading.Thread(target=audit), threading.Thread(target=writer)]
    for t in threads:
        t.start()

    assert write_done.wait(timeout=10), "核算期间写操作被阻塞（排队等待读连接/锁）"
    assert write_result["write_status"] == 201
    release.set()
    for t in threads:
        t.join(timeout=10)
    assert write_result.get("audit_status") == 200


def test_two_audits_run_in_parallel(client, monkeypatch):
    """两个审计请求的几何核算阶段可以重叠（读连接池提供独立连接）。"""
    from app.main import service

    _seed(client)

    active = 0
    lock = threading.Lock()
    overlapped = threading.Event()
    release = threading.Event()
    original = service._coverage_intervals

    def parallel_gated(zrow, points, lo, hi):
        nonlocal active
        with lock:
            active += 1
            if active >= 2:
                overlapped.set()
        assert release.wait(timeout=10), "并行门未释放"
        try:
            return original(zrow, points, lo, hi)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(service, "_coverage_intervals", parallel_gated)

    statuses = []

    def audit():
        r = client.post("/api/v1/zones/par/co-occupancy", json=_co_occupancy_body())
        statuses.append(r.status_code)

    threads = [threading.Thread(target=audit), threading.Thread(target=audit)]
    for t in threads:
        t.start()

    try:
        assert overlapped.wait(timeout=10), "两个审计未在核算阶段并行"
    finally:
        release.set()
        for t in threads:
            t.join(timeout=10)

    assert statuses == [200, 200]


def test_failed_reads_release_connections(client, monkeypatch):
    """读视图查询失败后连接归还：反复失败不影响后续读取（连接池不泄漏）。"""
    from app.main import db, service

    _seed(client)

    calls = {"n": 0}
    original = db._latest_zone_row

    def flaky(conn, zone_id):
        calls["n"] += 1
        raise RuntimeError("模拟读视图内查询失败")

    # 前若干次在快照内取区域行时失败（404/500 由异常处理器决定），随后恢复
    def maybe_fail(conn, zone_id):
        if calls["n"] < 30:
            return flaky(conn, zone_id)
        return original(conn, zone_id)

    monkeypatch.setattr(db, "_latest_zone_row", staticmethod(maybe_fail))

    for _ in range(30):
        with pytest.raises(RuntimeError):
            service.query_co_occupancy("par", _co_occupancy_body())

    monkeypatch.setattr(db, "_latest_zone_row", staticmethod(original))
    # 若读失败时连接未归还，池中连接耗尽后这里会永久等待
    result = service.query_co_occupancy("par", _co_occupancy_body())
    assert result["zone_id"] == "par"
