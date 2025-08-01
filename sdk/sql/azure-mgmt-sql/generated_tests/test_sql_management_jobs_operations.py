# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import pytest
from azure.mgmt.sql import SqlManagementClient

from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy

AZURE_LOCATION = "eastus"


@pytest.mark.skip("you may need to update the auto-generated test case before run it")
class TestSqlManagementJobsOperations(AzureMgmtRecordedTestCase):
    def setup_method(self, method):
        self.client = self.create_mgmt_client(SqlManagementClient)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_jobs_list_by_agent(self, resource_group):
        response = self.client.jobs.list_by_agent(
            resource_group_name=resource_group.name,
            server_name="str",
            job_agent_name="str",
            api_version="2024-11-01-preview",
        )
        result = [r for r in response]
        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_jobs_get(self, resource_group):
        response = self.client.jobs.get(
            resource_group_name=resource_group.name,
            server_name="str",
            job_agent_name="str",
            job_name="str",
            api_version="2024-11-01-preview",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_jobs_create_or_update(self, resource_group):
        response = self.client.jobs.create_or_update(
            resource_group_name=resource_group.name,
            server_name="str",
            job_agent_name="str",
            job_name="str",
            parameters={
                "description": "",
                "id": "str",
                "name": "str",
                "schedule": {
                    "enabled": bool,
                    "endTime": "9999-12-31T03:59:59-08:00",
                    "interval": "str",
                    "startTime": "0001-01-01T16:00:00-08:00",
                    "type": "Once",
                },
                "type": "str",
                "version": 0,
            },
            api_version="2024-11-01-preview",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_jobs_delete(self, resource_group):
        response = self.client.jobs.delete(
            resource_group_name=resource_group.name,
            server_name="str",
            job_agent_name="str",
            job_name="str",
            api_version="2024-11-01-preview",
        )

        # please add some check logic here by yourself
        # ...
