"""Tests for Docker, Kubernetes manifests, logging, and deployment endpoints."""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app import app, readiness, predict, PredictionInput

ROOT = Path(__file__).resolve().parent.parent


# ================================================================
# Dockerfile tests
# ================================================================
class TestDockerfiles:
    """Validate that required Docker files exist and are well-formed."""

    def test_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").exists()

    def test_dockerignore_exists(self):
        assert (ROOT / ".dockerignore").exists()

    def test_dockerfile_exposes_port_8000(self):
        content = (ROOT / "Dockerfile").read_text()
        assert "EXPOSE 8000" in content

    def test_dockerfile_uses_non_root_user(self):
        content = (ROOT / "Dockerfile").read_text()
        assert "USER" in content

    def test_dockerfile_has_healthcheck(self):
        content = (ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_dockerignore_excludes_venv(self):
        content = (ROOT / ".dockerignore").read_text()
        assert ".venv" in content

    def test_dockerignore_excludes_git(self):
        content = (ROOT / ".dockerignore").read_text()
        assert ".git" in content

    def test_dockerignore_excludes_data(self):
        content = (ROOT / ".dockerignore").read_text()
        assert "data/" in content


# ================================================================
# Kubernetes manifest fixtures & tests
# ================================================================
@pytest.fixture(scope="module")
def deployment_manifest():
    path = ROOT / "k8s" / "deployment.yaml"
    assert path.exists(), "k8s/deployment.yaml not found"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def service_manifest():
    path = ROOT / "k8s" / "service.yaml"
    assert path.exists(), "k8s/service.yaml not found"
    with open(path) as f:
        return yaml.safe_load(f)


class TestKubernetesManifests:
    """Validate Kubernetes YAML structure without requiring a live cluster."""

    def test_deployment_api_version(self, deployment_manifest):
        assert deployment_manifest["apiVersion"] == "apps/v1"

    def test_deployment_kind(self, deployment_manifest):
        assert deployment_manifest["kind"] == "Deployment"

    def test_deployment_name(self, deployment_manifest):
        assert deployment_manifest["metadata"]["name"] == "tea-leaf-api"

    def test_deployment_replicas(self, deployment_manifest):
        assert deployment_manifest["spec"]["replicas"] == 3

    def test_deployment_container_port(self, deployment_manifest):
        containers = deployment_manifest["spec"]["template"]["spec"]["containers"]
        ports = containers[0]["ports"]
        assert any(p["containerPort"] == 8000 for p in ports)

    def test_deployment_has_readiness_probe(self, deployment_manifest):
        container = deployment_manifest["spec"]["template"]["spec"]["containers"][0]
        assert "readinessProbe" in container
        assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
        assert container["readinessProbe"]["httpGet"]["port"] == 8000

    def test_deployment_has_liveness_probe(self, deployment_manifest):
        container = deployment_manifest["spec"]["template"]["spec"]["containers"][0]
        assert "livenessProbe" in container
        assert container["livenessProbe"]["httpGet"]["path"] == "/health"
        assert container["livenessProbe"]["httpGet"]["port"] == 8000

    def test_deployment_has_startup_probe(self, deployment_manifest):
        container = deployment_manifest["spec"]["template"]["spec"]["containers"][0]
        assert "startupProbe" in container
        assert container["startupProbe"]["httpGet"]["path"] == "/health"
        assert container["startupProbe"]["httpGet"]["port"] == 8000

    def test_deployment_has_resource_limits(self, deployment_manifest):
        container = deployment_manifest["spec"]["template"]["spec"]["containers"][0]
        assert "resources" in container
        assert "limits" in container["resources"]
        assert "requests" in container["resources"]

    def test_deployment_rolling_update_strategy(self, deployment_manifest):
        strategy = deployment_manifest["spec"]["strategy"]
        assert strategy["type"] == "RollingUpdate"

    def test_deployment_selector_matches_labels(self, deployment_manifest):
        selector = deployment_manifest["spec"]["selector"]["matchLabels"]
        template_labels = deployment_manifest["spec"]["template"]["metadata"]["labels"]
        for key, value in selector.items():
            assert template_labels.get(key) == value

    def test_service_api_version(self, service_manifest):
        assert service_manifest["apiVersion"] == "v1"

    def test_service_kind(self, service_manifest):
        assert service_manifest["kind"] == "Service"

    def test_service_name(self, service_manifest):
        assert service_manifest["metadata"]["name"] == "tea-leaf-api-service"

    def test_service_type(self, service_manifest):
        assert service_manifest["spec"]["type"] == "ClusterIP"

    def test_service_port(self, service_manifest):
        ports = service_manifest["spec"]["ports"]
        assert any(p["port"] == 80 for p in ports)

    def test_service_target_port(self, service_manifest):
        ports = service_manifest["spec"]["ports"]
        assert any(p["targetPort"] == 8000 for p in ports)

    def test_service_selector_matches_deployment(self, deployment_manifest, service_manifest):
        deploy_labels = deployment_manifest["spec"]["template"]["metadata"]["labels"]
        svc_selector = service_manifest["spec"]["selector"]
        for key, value in svc_selector.items():
            assert deploy_labels.get(key) == value


# ================================================================
# Readiness endpoint tests
# ================================================================
class TestReadinessEndpoint:
    """Verify readiness probe endpoint behavior."""

    def test_readiness_direct(self):
        response = readiness()
        assert response["status"] == "ready"
        assert response["model_loaded"] is True

    def test_readiness_http(self):
        client = TestClient(app)
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True


# ================================================================
# Logging tests
# ================================================================
class TestApplicationLogging:
    """Verify that the app configures and executes structured logging."""

    def test_app_uses_logging(self):
        content = (ROOT / "app.py").read_text()
        assert "import logging" in content

    def test_app_configures_logger(self):
        content = (ROOT / "app.py").read_text()
        assert "getLogger" in content

    def test_app_logs_to_stdout(self):
        content = (ROOT / "app.py").read_text()
        assert "StreamHandler" in content

    def test_app_has_lifespan_logging(self):
        content = (ROOT / "app.py").read_text()
        assert "starting up" in content.lower() or "startup" in content.lower()

    def test_app_has_prediction_logging(self):
        content = (ROOT / "app.py").read_text()
        assert "Prediction" in content

    def test_prediction_logs_execution(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="tea_leaf_api"):
            data = PredictionInput(
                worker_exp_yr=48,
                rain_mm_week=285,
                slope_deg=21,
                temp_c=15.9,
                bush_age_yr=10,
                hours_worked=5.4,
                flush="monsoon",
                section="River Side",
            )
            predict(data)

        log_messages = [record.message for record in caplog.records]
        assert any("Prediction request" in msg for msg in log_messages)
        assert any("Prediction successful" in msg for msg in log_messages)
