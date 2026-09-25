"""Real Flow HTTP writer, AI-worker 3C sender and Memory API/projector.

Explicit disposable opt-in only. Reuse the 4D process harness's loopback fence
and synthetic model/storage boundary; never substitute history, privacy,
application admission, event capture, delivery, or the reducer.
"""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import httpx
import psycopg
import pytest
from psycopg import sql

from tests.test_candidate_index_cross_system import ProcessStack

pytestmark = [
    pytest.mark.skipif(
        os.getenv("ACTIVEKG_HISTORY_XS_DISPOSABLE") != "1",
        reason="explicit disposable process proof not selected",
    ),
    pytest.mark.timeout(600),
]


class HistoryStack(ProcessStack):
    superuser_owner = False
    upgrade_from_4d = False
    defer_reference = False
    suppress_index = False

    def __init__(self, *args):
        super().__init__(*args)
        self.proxyport = self.port()
        self.ports.append(self.proxyport)

    def run(self, command, env, cwd=None, timeout=120):
        if self.upgrade_from_4d and str(command[-1]) == "scripts/init_railway_db.py":
            self.upgrade_from_4d = False
            baseline = self.path / "released-memory"
            baseline.mkdir()
            archive = subprocess.check_output(
                ["git", "archive", "d7e233a6f4f83adc7f4ade05110f6f96955240af"], cwd=self.memory
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
                bundle.extractall(baseline, filter="data")
            super().run(command, {**env, "PYTHONPATH": str(baseline)}, baseline, timeout)
            with psycopg.connect(env["ACTIVEKG_MIGRATE_DSN"], autocommit=True) as conn:
                query = "SELECT row_to_json(m) FROM schema_migrations m ORDER BY filename"
                before = conn.execute(query).fetchall()
                assert len(before) == 28
                upgraded = {**env, "ACTIVEKG_SCHEMA_FRESH_INIT": "0"}
                super().run(command, upgraded, cwd, timeout)
                after = conn.execute(query).fetchall()
                retained = [
                    row
                    for row in after
                    if row[0]["filename"] != "029_organization_candidate_history.sql"
                ]
                assert len(after) == 29 and retained == before
                for table in ("bindings", "event_state", "applications", "subjects", "scan_state"):
                    assert conn.execute(
                        sql.SQL("SELECT count(*) FROM {}").format(
                            sql.Identifier("organization_candidate_history_" + table)
                        )
                    ).fetchone() == (0,)
                super().run(command, upgraded, cwd, timeout)
                assert conn.execute(query).fetchall() == after
            return
        return super().run(command, env, cwd, timeout)

    def bootstrap(self):
        super().bootstrap()
        if self.superuser_owner:
            owner = self.owner.info.user
            assert owner in self.owned_roles and owner.endswith("_owner_test")
            # A second process proof models production's existing superuser
            # owner. Only a newly created disposable role can be elevated.
            self.admin.execute(sql.SQL("ALTER ROLE {} SUPERUSER").format(sql.Identifier(owner)))

    def start(self, command, env, cwd=None):
        if str(command[-1]) == "api":
            env = {**env, "ORG_CANDIDATE_HISTORY_ENABLED": "true"}
        if str(command[-1]).endswith("index.mjs"):
            env = dict(env)
            if self.defer_reference:
                env["ORGANIZATION_CANDIDATE_SYNC_ENABLED"] = "false"
            if self.suppress_index:
                env.pop("FLOW_CANDIDATE_INDEX_MODE", None)
        return super().start(command, env, cwd)

    def projection_lock(self, event, acquire):
        # Pause this event using the real reducer's nonblocking coordination
        # lock. The API stays enabled and reads must remain available; neither
        # the projector nor its repository is replaced by a test double.
        function = "pg_advisory_lock" if acquire else "pg_advisory_unlock"
        self.owner.execute(
            sql.SQL("SELECT {}(hashtextextended('candidate-history:'||%s,0))").format(
                sql.Identifier(function)
            ),
            (str(event),),
        )

    def start_sender(self):
        # Build and run the shipped AI-worker entrypoint, not an invocation of
        # the processor function. No AI jobs are queued by this rehearsal.
        output = self.path / "ai-worker.mjs"
        self.run(
            [
                self.flow / "node_modules/.bin/esbuild",
                "server/aiWorker.ts",
                "--platform=node",
                "--packages=external",
                "--bundle",
                "--format=esm",
                f"--outfile={output}",
            ],
            {},
        )
        worker_env = {
            key: value for key, value in self.fe.items() if key != "FLOW_CANDIDATE_INDEX_MODE"
        }
        self.sender = self.start(
            ["node", "--import", self.flow_fence, output],
            {
                **worker_env,
                "ORGANIZATION_CANDIDATE_SYNC_ENABLED": "false",
                "CANDIDATE_CONSENT_DELIVERY_ENABLED": "false",
                "DECISION_PROJECTION_DELIVERY_ENABLED": "true",
            },
        )

    def stop_sender(self):
        self.stop(self.sender)
        self.retired = [*getattr(self, "retired", []), self.sender]

    def history(self, client, application):
        response = client.get(f"/api/applications/{application}/decision-history")
        assert response.headers.get("cache-control") == "private, no-store"
        return response


class MemoryResponseBarrier:
    """Forward the real receiver bytes; mutate only disposable Flow state after
    the receiver responds and before Flow receives it. Never fake a receiver.
    """

    def __init__(self, stack):
        self.callback = None
        self.calls = 0
        barrier = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                assert self.path.startswith("/") and not self.path.startswith("//")
                body = self.rfile.read(int(self.headers.get("content-length", "0")))
                with httpx.Client(trust_env=False, timeout=10) as client:
                    reply = client.request(
                        self.command,
                        stack.api_url + self.path,
                        content=body,
                        headers={
                            "authorization": self.headers.get("authorization", ""),
                            "content-type": "application/json",
                        },
                    )
                if self.path == "/organization-candidate-history/read":
                    barrier.calls += 1
                    callback, barrier.callback = barrier.callback, None
                    if callback:
                        callback()
                self.send_response(reply.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply.content)))
                self.end_headers()
                self.wfile.write(reply.content)

            do_GET = do_POST

        self.server = ThreadingHTTPServer(("127.0.0.1", stack.proxyport), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()


def test_exact_released_memory_28_to_29_and_idempotent_ledger(tmp_path):
    stack = HistoryStack(tmp_path, None)
    stack.upgrade_from_4d = True
    try:
        stack.bootstrap()
    finally:
        stack.close()


def test_real_event_waits_for_reference_then_source_and_restarts_without_recount(tmp_path):
    stack = HistoryStack(tmp_path, None)
    stack.defer_reference = True
    try:
        stack.setup()
        with (
            psycopg.connect(stack.fo, autocommit=True) as flow,
            httpx.Client(base_url=stack.web_url, trust_env=False, timeout=15) as lead,
            httpx.Client(base_url=stack.web_url, trust_env=False, timeout=15) as candidate,
        ):
            org, recruiter, email, job = stack.seed_job(flow)
            _, applicant_email = stack.seed_user(flow, "candidate")
            flow.execute(
                'INSERT INTO pipeline_stages(organization_id,name,"order",is_default,created_by) '
                "VALUES(%s,'Applied',0,true,%s)",
                (org, recruiter),
            )
            stage = flow.execute(
                'INSERT INTO pipeline_stages(organization_id,name,"order",is_default,created_by) '
                "VALUES(%s,'Review',1,false,%s) RETURNING id",
                (org, recruiter),
            ).fetchone()[0]
            stack.privacy_fresh(flow)
            stack.login(lead, email)
            stack.login(candidate, applicant_email)
            assert stack.apply(candidate, job, applicant_email).status_code in (200, 201)
            app = flow.execute("SELECT id FROM applications WHERE job_id=%s", (job,)).fetchone()[0]
            assert (
                lead.patch(
                    f"/api/applications/{app}/stage",
                    headers=stack.csrf(lead),
                    json={"stageId": stage},
                ).status_code
                == 200
            )
            event = flow.execute(
                "SELECT event_id FROM decision_events WHERE organization_id=%s AND aggregate_id=%s",
                (org, app),
            ).fetchone()[0]
            stack.start_sender()

            def state(expected):
                return stack.owner.execute(
                    "SELECT state FROM organization_candidate_history_event_state WHERE event_id=%s",
                    (event,),
                ).fetchone() == (expected,)

            stack.wait(lambda: state("waiting_reference"))
            assert stack.owner.execute(
                "SELECT count(*) FROM organization_candidate_history_bindings WHERE event_id=%s",
                (event,),
            ).fetchone() == (0,)
            stack.defer_reference = False
            stack.suppress_index = True
            stack.restart_web()
            stack.wait(lambda: state("waiting_source"))
            stack.suppress_index = False
            stack.restart_web()
            stack.published(flow, app)
            stack.wait(lambda: state("applied"))

            def caught_up():
                response = stack.history(lead, app)
                if response.status_code == 503:
                    assert response.json()["summary"] is None
                    return False
                assert response.status_code == 200
                body = response.json()
                if body["freshness"]["status"] != "caught_up_to_observed_capture":
                    return False
                assert body["summary"]["observed_stage_move_count"] == "1"
                return True

            stack.wait(caught_up)
            stack.stop_sender()
            stack.start_sender()
            stack.restart_web()
            stack.wait(caught_up)
            assert stack.owner.execute(
                "SELECT count(*) FROM organization_candidate_history_bindings WHERE event_id=%s",
                (event,),
            ).fetchone() == (1,)
            assert stack.forbidden.read_text() == ""
    finally:
        stack.close()


@pytest.mark.parametrize("superuser_owner", [False, True], ids=["restricted-owner", "super-owner"])
@pytest.mark.parametrize(
    "global_optout", [False, True], ids=["ordinary-applicant", "global-optout"]
)
def test_real_stage_delivery_private_optout_history_and_erasure(
    tmp_path, superuser_owner, global_optout
):
    stack = HistoryStack(tmp_path, None)
    stack.superuser_owner = superuser_owner
    try:
        stack.setup()
        with (
            psycopg.connect(stack.fo, autocommit=True) as flow,
            httpx.Client(base_url=stack.web_url, trust_env=False, timeout=15) as lead,
            httpx.Client(base_url=stack.web_url, trust_env=False, timeout=15) as candidate,
            httpx.Client(base_url=stack.web_url, trust_env=False, timeout=15) as stranger,
        ):
            org, recruiter, email, job = stack.seed_job(flow)
            _, candidate_email = stack.seed_user(flow, "candidate")
            _, _, stranger_email, _ = stack.seed_job(flow)
            stages = [
                flow.execute(
                    'INSERT INTO pipeline_stages(organization_id,name,"order",is_default,created_by) '
                    "VALUES(%s,%s,%s,%s,%s) RETURNING id",
                    (org, name, position, position == 0, recruiter),
                ).fetchone()[0]
                for position, name in enumerate(("Applied", "Review", "Interview"))
            ]
            stack.privacy_fresh(flow)
            stack.login(lead, email)
            stack.login(candidate, candidate_email)
            stack.login(stranger, stranger_email)

            def privacy(action, decision):
                assert (
                    stack.post(
                        candidate,
                        "/api/candidate/privacy/reauth",
                        json={"password": "fixture-password"},
                    ).status_code
                    == 200
                )
                request = str(uuid4())
                assert (
                    stack.post(
                        candidate,
                        "/api/candidate/privacy/requests",
                        json={"requestId": request, "action": action},
                    ).status_code
                    == 202
                )

                def delivered():
                    response = candidate.get("/api/candidate/privacy/status")
                    assert response.status_code == 200
                    return any(
                        row["requestId"] == request
                        and row["deliveryStatus"] == "delivered"
                        and row["decision"] == decision
                        for row in response.json()["requests"]
                    )

                stack.wait(delivered)

            # Positive control and D14 run through the identical application
            # and history path; only the real prior opt-out request differs.
            if global_optout:
                privacy("withdraw_global_matching", "block_global")
            applied = stack.apply(candidate, job, candidate_email)
            assert applied.status_code in (200, 201), applied.status_code
            application = flow.execute(
                "SELECT id FROM applications WHERE job_id=%s AND email=%s",
                (job, candidate_email),
            ).fetchone()[0]
            stack.published(flow, application)

            def move(stage):
                result = lead.patch(
                    f"/api/applications/{application}/stage",
                    headers=stack.csrf(lead),
                    json={"stageId": stage},
                )
                assert result.status_code == 200, result.status_code

            move(stages[1])
            # Sender has not started: Flow cannot claim delivery or a summary.
            waiting = stack.history(lead, application)
            assert waiting.status_code == 200, waiting.text
            assert waiting.json()["freshness"]["status"] == "awaiting_delivery"
            assert waiting.json()["summary"] is None
            event = flow.execute(
                "SELECT event_id FROM decision_events WHERE organization_id=%s "
                "AND aggregate_type='application' AND aggregate_id=%s",
                (org, application),
            ).fetchone()[0]
            stack.projection_lock(event, True)
            stack.start_sender()

            def projection_pending():
                response = stack.history(lead, application)
                if response.status_code == 503:
                    assert response.json() == {"code": "temporarily_unavailable", "summary": None}
                    return False
                assert response.status_code == 200, response.text
                return response.json()["freshness"]["status"] == "projection_pending"

            stack.wait(projection_pending)
            stack.projection_lock(event, False)

            def caught_up(count):
                response = stack.history(lead, application)
                if response.status_code == 503:
                    # Real projector/source privacy-lock overlap is explicitly
                    # retryable. Require the closed empty body, then eventual
                    # success inside the existing bounded wait; never accept
                    # stale data or suppress another denial/error shape.
                    assert response.json() == {"code": "temporarily_unavailable", "summary": None}
                    return False
                assert response.status_code == 200, response.text
                body = response.json()
                if body["freshness"]["status"] != "caught_up_to_observed_capture":
                    return False
                assert body["authority_status"] == "eligible"
                assert body["summary"]["observed_stage_move_count"] == str(count)
                return body

            first = stack.wait(lambda: caught_up(1))
            assert first["summary"]["latest_observed_stage_id"] == stages[1]
            assert stack.history(stranger, application).status_code == 404
            assert stack.history(candidate, application).status_code == 404
            flow.execute(
                "UPDATE organization_members SET seat_assigned=false "
                "WHERE organization_id=%s AND user_id=%s",
                (org, recruiter),
            )
            try:
                assert stack.history(lead, application).status_code == 404
            finally:
                flow.execute(
                    "UPDATE organization_members SET seat_assigned=true "
                    "WHERE organization_id=%s AND user_id=%s",
                    (org, recruiter),
                )

            stack.stop_sender()
            move(stages[2])
            waiting = stack.history(lead, application).json()
            assert waiting["freshness"]["status"] == "awaiting_delivery"
            stack.start_sender()
            second = stack.wait(lambda: caught_up(2))
            assert second["summary"]["first"] == first["summary"]["first"]
            assert second["summary"]["latest_observed_stage_id"] == stages[2]

            # A platform administrator has no shortcut into this private slice.
            with httpx.Client(base_url=stack.web_url, trust_env=False, timeout=15) as admin:
                _, admin_email = stack.seed_user(flow, "super_admin")
                stack.login(admin, admin_email)
                assert stack.history(admin, application).status_code == 404

            with httpx.Client(base_url=stack.web_url, trust_env=False, timeout=15) as colleague:
                member, member_email = stack.seed_user(flow, "recruiter")
                flow.execute(
                    "INSERT INTO organization_members(organization_id,user_id,role,seat_assigned) "
                    "VALUES(%s,%s,'member',true)",
                    (org, member),
                )
                flow.execute(
                    "INSERT INTO job_recruiters(organization_id,job_id,recruiter_id,added_by) "
                    "VALUES(%s,%s,%s,%s)",
                    (org, job, member, recruiter),
                )
                stack.login(colleague, member_email)
                stack.wait(lambda: stack.history(colleague, application).status_code == 200)
                flow.execute(
                    "DELETE FROM job_recruiters WHERE job_id=%s AND recruiter_id=%s", (job, member)
                )
                assert stack.history(colleague, application).status_code == 404

            with MemoryResponseBarrier(stack) as barrier:
                stack.restart_web(ACTIVEKG_BASE_URL=f"http://127.0.0.1:{stack.proxyport}")
                barrier.callback = lambda: flow.execute(
                    "UPDATE organization_members SET seat_assigned=false WHERE organization_id=%s AND user_id=%s",
                    (org, recruiter),
                )
                assert stack.history(lead, application).status_code == 404
                flow.execute(
                    "UPDATE organization_members SET seat_assigned=true WHERE organization_id=%s AND user_id=%s",
                    (org, recruiter),
                )
                calls = barrier.calls
                barrier.callback = lambda: move(stages[1])
                raced = stack.history(lead, application)
                assert 1 <= barrier.calls - calls <= 2
                if raced.status_code == 200:
                    assert raced.json()["freshness"]["expected"]["count"] == "3"
                else:
                    assert raced.status_code in (409, 503) and raced.json()["summary"] is None
                third = stack.wait(lambda: caught_up(3))
                assert third["summary"]["latest_observed_stage_id"] == stages[1]
                stack.restart_web(ACTIVEKG_BASE_URL=stack.api_url)

            # Identical synthetic contact details do not authorize cross-job
            # aggregation. The second real application gets its own 4B source.
            _, _, _, other_job = stack.seed_job(flow, org=org, recruiter=recruiter)
            assert stack.apply(candidate, other_job, candidate_email).status_code in (200, 201)
            other_app = flow.execute(
                "SELECT id FROM applications WHERE job_id=%s AND email=%s",
                (other_job, candidate_email),
            ).fetchone()[0]
            stack.published(flow, other_app)
            assert (
                lead.patch(
                    f"/api/applications/{other_app}/stage",
                    headers=stack.csrf(lead),
                    json={"stageId": stages[1]},
                ).status_code
                == 200
            )

            def other_caught_up():
                reply = stack.history(lead, other_app)
                if reply.status_code == 503:
                    assert reply.json()["summary"] is None
                    return False
                assert reply.status_code == 200
                value = reply.json()
                return (
                    value
                    if value["freshness"]["status"] == "caught_up_to_observed_capture"
                    else False
                )

            other_history = stack.wait(other_caught_up)
            assert other_history["summary"]["observed_stage_move_count"] == "1"
            assert other_history["binding"]["reference_id"] != third["binding"]["reference_id"]
            assert caught_up(3)

            # Erasure uses the real request path and actual privacy processor.
            privacy("request_erasure", "block_all")
            restricted = stack.history(lead, application)
            # The existing Flow authorization fence hides erased applications
            # before any Memory call, preserving the non-enumerating 404.
            assert restricted.status_code == 404, restricted.text
            assert restricted.json() == {"code": "not_found"}
            assert stack.forbidden.read_text() == ""
    finally:
        stack.close()
