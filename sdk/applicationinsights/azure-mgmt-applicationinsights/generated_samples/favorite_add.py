# pylint: disable=line-too-long,useless-suppression
# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential

from azure.mgmt.applicationinsights import ApplicationInsightsManagementClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-applicationinsights
# USAGE
    python favorite_add.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = ApplicationInsightsManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id="subid",
    )

    response = client.favorites.add(
        resource_group_name="my-resource-group",
        resource_name="my-ai-component",
        favorite_id="deadb33f-8bee-4d3b-a059-9be8dac93960",
        favorite_properties={
            "Category": None,
            "Config": '{"MEDataModelRawJSON":"{\\n  \\"version\\": \\"1.4.1\\",\\n  \\"isCustomDataModel\\": true,\\n  \\"items\\": [\\n    {\\n      \\"id\\": \\"90a7134d-9a38-4c25-88d3-a495209873eb\\",\\n      \\"chartType\\": \\"Area\\",\\n      \\"chartHeight\\": 4,\\n      \\"metrics\\": [\\n        {\\n          \\"id\\": \\"preview/requests/count\\",\\n          \\"metricAggregation\\": \\"Sum\\",\\n          \\"color\\": \\"msportalfx-bgcolor-d0\\"\\n        }\\n      ],\\n      \\"priorPeriod\\": false,\\n      \\"clickAction\\": {\\n        \\"defaultBlade\\": \\"SearchBlade\\"\\n      },\\n      \\"horizontalBars\\": true,\\n      \\"showOther\\": true,\\n      \\"aggregation\\": \\"Sum\\",\\n      \\"percentage\\": false,\\n      \\"palette\\": \\"fail\\",\\n      \\"yAxisOption\\": 0,\\n      \\"title\\": \\"\\"\\n    },\\n    {\\n      \\"id\\": \\"0c289098-88e8-4010-b212-546815cddf70\\",\\n      \\"chartType\\": \\"Area\\",\\n      \\"chartHeight\\": 2,\\n      \\"metrics\\": [\\n        {\\n          \\"id\\": \\"preview/requests/duration\\",\\n          \\"metricAggregation\\": \\"Avg\\",\\n          \\"color\\": \\"msportalfx-bgcolor-j1\\"\\n        }\\n      ],\\n      \\"priorPeriod\\": false,\\n      \\"clickAction\\": {\\n        \\"defaultBlade\\": \\"SearchBlade\\"\\n      },\\n      \\"horizontalBars\\": true,\\n      \\"showOther\\": true,\\n      \\"aggregation\\": \\"Avg\\",\\n      \\"percentage\\": false,\\n      \\"palette\\": \\"greenHues\\",\\n      \\"yAxisOption\\": 0,\\n      \\"title\\": \\"\\"\\n    },\\n    {\\n      \\"id\\": \\"cbdaab6f-a808-4f71-aca5-b3976cbb7345\\",\\n      \\"chartType\\": \\"Bar\\",\\n      \\"chartHeight\\": 4,\\n      \\"metrics\\": [\\n        {\\n          \\"id\\": \\"preview/requests/duration\\",\\n          \\"metricAggregation\\": \\"Avg\\",\\n          \\"color\\": \\"msportalfx-bgcolor-d0\\"\\n        }\\n      ],\\n      \\"priorPeriod\\": false,\\n      \\"clickAction\\": {\\n        \\"defaultBlade\\": \\"SearchBlade\\"\\n      },\\n      \\"horizontalBars\\": true,\\n      \\"showOther\\": true,\\n      \\"aggregation\\": \\"Avg\\",\\n      \\"percentage\\": false,\\n      \\"palette\\": \\"magentaHues\\",\\n      \\"yAxisOption\\": 0,\\n      \\"title\\": \\"\\"\\n    },\\n    {\\n      \\"id\\": \\"1d5a6a3a-9fa1-4099-9cf9-05eff72d1b02\\",\\n      \\"grouping\\": {\\n        \\"kind\\": \\"ByDimension\\",\\n        \\"dimension\\": \\"context.application.version\\"\\n      },\\n      \\"chartType\\": \\"Grid\\",\\n      \\"chartHeight\\": 1,\\n      \\"metrics\\": [\\n        {\\n          \\"id\\": \\"basicException.count\\",\\n          \\"metricAggregation\\": \\"Sum\\",\\n          \\"color\\": \\"msportalfx-bgcolor-g0\\"\\n        },\\n        {\\n          \\"id\\": \\"requestFailed.count\\",\\n          \\"metricAggregation\\": \\"Sum\\",\\n          \\"color\\": \\"msportalfx-bgcolor-f0s2\\"\\n        }\\n      ],\\n      \\"priorPeriod\\": true,\\n      \\"clickAction\\": {\\n        \\"defaultBlade\\": \\"SearchBlade\\"\\n      },\\n      \\"horizontalBars\\": true,\\n      \\"showOther\\": true,\\n      \\"percentage\\": false,\\n      \\"palette\\": \\"blueHues\\",\\n      \\"yAxisOption\\": 0,\\n      \\"title\\": \\"\\"\\n    }\\n  ],\\n  \\"currentFilter\\": {\\n    \\"eventTypes\\": [\\n      1,\\n      2\\n    ],\\n    \\"typeFacets\\": {},\\n    \\"isPermissive\\": false\\n  },\\n  \\"timeContext\\": {\\n    \\"durationMs\\": 75600000,\\n    \\"endTime\\": \\"2018-01-31T20:30:00.000Z\\",\\n    \\"createdTime\\": \\"2018-01-31T23:54:26.280Z\\",\\n    \\"isInitialTime\\": false,\\n    \\"grain\\": 1,\\n    \\"useDashboardTimeRange\\": false\\n  },\\n  \\"jsonUri\\": \\"Favorite_BlankChart\\",\\n  \\"timeSource\\": 0\\n}"}',
            "FavoriteId": "deadb33f-8bee-4d3b-a059-9be8dac93960",
            "FavoriteType": "shared",
            "IsGeneratedFromTemplate": False,
            "Name": "Blah Blah Blah",
            "SourceType": None,
            "Tags": ["TagSample01", "TagSample02"],
            "TimeModified": None,
            "Version": "ME",
        },
    )
    print(response)


# x-ms-original-file: specification/applicationinsights/resource-manager/Microsoft.Insights/stable/2015-05-01/examples/FavoriteAdd.json
if __name__ == "__main__":
    main()
