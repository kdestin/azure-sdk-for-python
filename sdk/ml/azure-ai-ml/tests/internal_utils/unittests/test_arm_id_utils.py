import pytest

from azure.ai.ml._scope_dependent_operations import OperationScope
from azure.ai.ml._utils._arm_id_utils import get_arm_id_with_version, parse_name_label, parse_prefixed_name_label


@pytest.mark.unittest
@pytest.mark.core_sdk_test
def test_get_arm_id(mock_workspace_scope: OperationScope) -> None:
    arm_id = get_arm_id_with_version(mock_workspace_scope, "models", "modeltest", "2")
    expected_arm_id = (
        "/subscriptions/test_subscription/resourceGroups/test_resource_group"
        "/providers/Microsoft.MachineLearningServices/workspaces/test_workspace_name/models/modeltest/versions/2"
    )
    assert expected_arm_id == arm_id


@pytest.mark.unittest
def test_parse_name_label() -> None:
    assert parse_name_label("name") == ("name", None), "Should return original string and no label"
    assert parse_name_label("name:1") == ("name:1", None), "Should return versioned id and no label"
    assert parse_name_label("azureml:name:1") == ("azureml:name:1", None), "Should return prefixed id and no label"

    assert parse_name_label("name@latest") == ("name", "latest"), "Should parse name and label"
    assert parse_name_label("name@1") == ("name", "1"), "Should parse name and label"
    assert parse_name_label("azureml:name@latest") == ("azureml:name", "latest"), "Should parse label and keep prefix"


@pytest.mark.unittest
def test_parse_prefixed_name_label() -> None:
    assert parse_prefixed_name_label("name") == ("name", None), "Should return original string and no label"
    assert parse_prefixed_name_label("name:1") == ("name:1", None), "Should return versioned id and no label"
    assert parse_prefixed_name_label("azureml:name:1") == ("azureml:name:1", None), "Should return id and no label"

    assert parse_prefixed_name_label("name@latest") == ("name", "latest"), "Should parse name and label"
    assert parse_prefixed_name_label("name@1") == ("name", "1"), "Should parse name and label"

    assert parse_prefixed_name_label("azureml:name@latest") == ("name", "latest"), "Should parse name and label"
