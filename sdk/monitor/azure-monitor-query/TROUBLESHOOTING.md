# Troubleshooting Azure Monitor Query client library issues

This troubleshooting guide contains instructions to diagnose frequently encountered issues while using the Azure Monitor Query client library for Python.

## Table of contents

* [General Troubleshooting](#general-troubleshooting)
    * [Enable client logging](#enable-client-logging)
    * [Troubleshooting authentication issues with query requests](#authentication-errors)
    * [Troubleshooting running async APIs](#errors-with-running-async-apis)
* [Troubleshooting Logs Query](#troubleshooting-logs-query)
    * [Troubleshooting insufficient access error](#troubleshooting-insufficient-access-error-for-logs-query)
    * [Troubleshooting invalid Kusto query](#troubleshooting-invalid-kusto-query)
    * [Troubleshooting empty log query results](#troubleshooting-empty-log-query-results)
    * [Troubleshooting server timeouts when executing logs query request](#troubleshooting-server-timeouts-when-executing-logs-query-request)
    * [Troubleshooting partially successful logs query requests](#troubleshooting-partially-successful-logs-query-requests)
* [Additional azure-core configurations](#additional-azure-core-configurations)

## General Troubleshooting

Monitor query raises exceptions described in [`azure-core`](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core/README.md)

### Enable client logging

To troubleshoot issues with Azure Monitor query library, it is important to first enable logging to monitor the behavior of the application. The errors and warnings in the logs generally provide useful insights into what went wrong and sometimes include corrective actions to fix issues.

This library uses the standard [logging](https://docs.python.org/3/library/logging.html) library for logging. Basic information about HTTP sessions, such as URLs and headers, is logged at the INFO level.
Detailed DEBUG level logging, including request/response bodies and unredacted headers, can be enabled on a client with the logging_enable argument:

```python
import logging
from azure.monitor.query import LogsQueryClient

# Create a logger for the 'azure.monitor.query' SDK
logger = logging.getLogger('azure.monitor.query')
logger.setLevel(logging.DEBUG)

# Configure a console output
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)

client = LogsQueryClient(credential, logging_enable=True)
```

Similarly, logging_enable can enable detailed logging for a single operation, even when it isn't enabled for the client:

```python
client.query_workspace(logging_enable=True)
```

### Authentication errors

Azure Monitor Query supports Microsoft Entra ID authentication. LogsQueryClient has methods to set the `credential`. To provide a valid credential, you can use the `azure-identity` package. For more details on getting started, refer to the [README](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/monitor/azure-monitor-query#create-the-client) of the Azure Monitor Query library. You can also refer to
the [Azure Identity documentation](https://learn.microsoft.com/python/api/overview/azure/identity-readme)
for more details on the various types of credential supported in `azure-identity`.

For more help on troubleshooting authentication errors please see the Azure Identity client library [troubleshooting guide](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/identity/azure-identity/TROUBLESHOOTING.md).

### Errors with running async APIs

The async transport is designed to be opt-in. [AioHttp](https://pypi.org/project/aiohttp/) is one of the supported implementations of async transport. It is not installed by default. You need to install it separately as follows:

```
pip install aiohttp
```

## Troubleshooting Logs Query

### Troubleshooting insufficient access error for logs query

If you get an HTTP error with status code 403 (Forbidden), it means that the provided credentials does not have
sufficient permissions to query the workspace.
```text
"{"error":{"message":"The provided credentials have insufficient access to perform the requested operation","code":"InsufficientAccessError","correlationId":""}}"
```

1. Check that the application or user that is making the request has sufficient permissions:
    * You can refer to this document to [manage access to workspaces](https://learn.microsoft.com/azure/azure-monitor/logs/manage-access#manage-access-using-workspace-permissions)
2. If the user or application is granted sufficient privileges to query the workspace, make sure you are
   authenticating as that user/application. If you are authenticating using the
   [DefaultAzureCredential](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/identity/azure-identity#defaultazurecredential)
   then check the logs to verify that the credential used is the one you expected. To enable logging, see [enable
   client logging](#enable-client-logging) section above.

For more help on troubleshooting authentication errors please see the Azure Identity client library [troubleshooting guide](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/identity/azure-identity/TROUBLESHOOTING.md).

### Troubleshooting invalid Kusto query

If you get an HTTP error with status code 400 (Bad Request), you may have an error in your Kusto query and you'll
see an error message similar to the one below.

```text
(BadArgumentError) The request had some invalid properties
Code: BadArgumentError
Message: The request had some invalid properties
Inner error: {
    "code": "SemanticError",
    "message": "A semantic error occurred.",
    "innererror": {
        "code": "SEM0100",
        "message": "'take' operator: Failed to resolve table or column expression named 'Appquests'"
    }
}
```

The error message in the innererror may include the where the Kusto query has an error. You may also refer to the [Kusto Query Language](https://learn.microsoft.com/azure/data-explorer/kusto/query) reference docs to learn more about querying logs using KQL.

### Troubleshooting empty log query results

If your Kusto query returns empty no logs, please validate the following:

- You have the right workspace ID
- You are setting the correct time interval for the query. Try expanding the time interval for your query to see if that
  returns any results.
- If your Kusto query also has a time interval, the query is evaluated for the intersection of the time interval in the
  query string and the time interval set in the `timespan` param provided the query API. The intersection of
  these time intervals may not have any logs. To avoid any confusion, it's recommended to remove any time interval in
  the Kusto query string and use `timespan` explicitly. Please note that the `timespan` param can be a timedelta,
  a timedelta and a start datetime, or a start datetime/end datetime.

### Troubleshooting server timeouts when executing logs query request

Some complex Kusto queries can take a long time to complete and such queries are aborted by the
service if they run for more than 3 minutes. For such scenarios, the query APIs on `LogsQueryClient`, provide options to
configure the timeout on the server. The server timeout can be extended up to 10 minutes.

You may see an error as follows:

```text
Code: GatewayTimeout
Message: Gateway timeout
Inner error: {
    "code": "GatewayTimeout",
    "message": "Unable to unzip response"
}
```

The following code shows a sample on how to set the server timeout to 10 minutes. Note that by setting this server
timeout, the Azure Monitor Query library will automatically also extend the client timeout to wait for 10 minutes for
the server to respond. You don't need to configure your HTTP client to extend the response timeout as shown in the
previous section.

```python
from azure.monitor.query import LogsQueryClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()

client = LogsQueryClient(credential)

client.query_workspace(
    "{workspaceId}",
    "{kusto-query-string}",
    timespan="{timespan}",
    server_timeout=600)
```

### Troubleshooting partially successful logs query requests

If the execution of a Kusto query resulted in a partially successful response, the Azure Monitor Query
client library will return `partial_data` and `partial_error` indicating that the query was not fully successful.

```python
response = client.query_workspace("{workspaceId}", "{kusto-query-string}", timespan="{timespan}")

data = response.partial_data
error = response.partial_error
```

## Additional azure-core configurations

When calling the methods, some properties including `retry_mode`, `timeout`, `connection_verify` can be configured by passing in as keyword arguments. See
[configurations](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core/README.md#configurations) for list of all such properties.
