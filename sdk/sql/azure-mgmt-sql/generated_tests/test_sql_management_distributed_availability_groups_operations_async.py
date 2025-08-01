# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import pytest
from azure.mgmt.sql.aio import SqlManagementClient

from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer
from devtools_testutils.aio import recorded_by_proxy_async

AZURE_LOCATION = "eastus"


@pytest.mark.skip("you may need to update the auto-generated test case before run it")
class TestSqlManagementDistributedAvailabilityGroupsOperationsAsync(AzureMgmtRecordedTestCase):
    def setup_method(self, method):
        self.client = self.create_mgmt_client(SqlManagementClient, is_async=True)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_distributed_availability_groups_list_by_instance(self, resource_group):
        response = self.client.distributed_availability_groups.list_by_instance(
            resource_group_name=resource_group.name,
            managed_instance_name="str",
            api_version="2024-11-01-preview",
        )
        result = [r async for r in response]
        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_distributed_availability_groups_get(self, resource_group):
        response = await self.client.distributed_availability_groups.get(
            resource_group_name=resource_group.name,
            managed_instance_name="str",
            distributed_availability_group_name="str",
            api_version="2024-11-01-preview",
        )

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_distributed_availability_groups_begin_create_or_update(self, resource_group):
        response = await (
            await self.client.distributed_availability_groups.begin_create_or_update(
                resource_group_name=resource_group.name,
                managed_instance_name="str",
                distributed_availability_group_name="str",
                parameters={
                    "databases": [
                        {
                            "connectedState": "str",
                            "databaseName": "str",
                            "instanceRedoReplicationLagSeconds": 0,
                            "instanceReplicaId": "str",
                            "instanceSendReplicationLagSeconds": 0,
                            "lastBackupLsn": "str",
                            "lastBackupTime": "2020-02-20 00:00:00",
                            "lastCommitLsn": "str",
                            "lastCommitTime": "2020-02-20 00:00:00",
                            "lastHardenedLsn": "str",
                            "lastHardenedTime": "2020-02-20 00:00:00",
                            "lastReceivedLsn": "str",
                            "lastReceivedTime": "2020-02-20 00:00:00",
                            "lastSentLsn": "str",
                            "lastSentTime": "2020-02-20 00:00:00",
                            "mostRecentLinkError": "str",
                            "partnerAuthCertValidity": {"certificateName": "str", "expiryDate": "2020-02-20 00:00:00"},
                            "partnerReplicaId": "str",
                            "replicaState": "str",
                            "seedingProgress": "str",
                            "synchronizationHealth": "str",
                        }
                    ],
                    "distributedAvailabilityGroupId": "str",
                    "distributedAvailabilityGroupName": "str",
                    "failoverMode": "str",
                    "id": "str",
                    "instanceAvailabilityGroupName": "str",
                    "instanceLinkRole": "str",
                    "name": "str",
                    "partnerAvailabilityGroupName": "str",
                    "partnerEndpoint": "str",
                    "partnerLinkRole": "str",
                    "replicationMode": "str",
                    "seedingMode": "str",
                    "type": "str",
                },
                api_version="2024-11-01-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_distributed_availability_groups_begin_delete(self, resource_group):
        response = await (
            await self.client.distributed_availability_groups.begin_delete(
                resource_group_name=resource_group.name,
                managed_instance_name="str",
                distributed_availability_group_name="str",
                api_version="2024-11-01-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_distributed_availability_groups_begin_update(self, resource_group):
        response = await (
            await self.client.distributed_availability_groups.begin_update(
                resource_group_name=resource_group.name,
                managed_instance_name="str",
                distributed_availability_group_name="str",
                parameters={
                    "databases": [
                        {
                            "connectedState": "str",
                            "databaseName": "str",
                            "instanceRedoReplicationLagSeconds": 0,
                            "instanceReplicaId": "str",
                            "instanceSendReplicationLagSeconds": 0,
                            "lastBackupLsn": "str",
                            "lastBackupTime": "2020-02-20 00:00:00",
                            "lastCommitLsn": "str",
                            "lastCommitTime": "2020-02-20 00:00:00",
                            "lastHardenedLsn": "str",
                            "lastHardenedTime": "2020-02-20 00:00:00",
                            "lastReceivedLsn": "str",
                            "lastReceivedTime": "2020-02-20 00:00:00",
                            "lastSentLsn": "str",
                            "lastSentTime": "2020-02-20 00:00:00",
                            "mostRecentLinkError": "str",
                            "partnerAuthCertValidity": {"certificateName": "str", "expiryDate": "2020-02-20 00:00:00"},
                            "partnerReplicaId": "str",
                            "replicaState": "str",
                            "seedingProgress": "str",
                            "synchronizationHealth": "str",
                        }
                    ],
                    "distributedAvailabilityGroupId": "str",
                    "distributedAvailabilityGroupName": "str",
                    "failoverMode": "str",
                    "id": "str",
                    "instanceAvailabilityGroupName": "str",
                    "instanceLinkRole": "str",
                    "name": "str",
                    "partnerAvailabilityGroupName": "str",
                    "partnerEndpoint": "str",
                    "partnerLinkRole": "str",
                    "replicationMode": "str",
                    "seedingMode": "str",
                    "type": "str",
                },
                api_version="2024-11-01-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_distributed_availability_groups_begin_failover(self, resource_group):
        response = await (
            await self.client.distributed_availability_groups.begin_failover(
                resource_group_name=resource_group.name,
                managed_instance_name="str",
                distributed_availability_group_name="str",
                parameters={"failoverType": "str"},
                api_version="2024-11-01-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy_async
    async def test_distributed_availability_groups_begin_set_role(self, resource_group):
        response = await (
            await self.client.distributed_availability_groups.begin_set_role(
                resource_group_name=resource_group.name,
                managed_instance_name="str",
                distributed_availability_group_name="str",
                parameters={"instanceRole": "str", "roleChangeType": "str"},
                api_version="2024-11-01-preview",
            )
        ).result()  # call '.result()' to poll until service return final result

        # please add some check logic here by yourself
        # ...
