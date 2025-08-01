---
page_type: sample
languages:
  - python
products:
  - azure
  - azure-event-hubs
urlFragment: eventhub-samples
---

# Azure Event Hubs client library for Python Samples

These are code samples that show common scenario operations with the Azure Event Hubs client library.
Both [sync version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples) and [async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples) of samples are provided.

- [client_creation.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/client_creation.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/client_creation_async.py)) - Examples to create EventHubProducerClient and EventHubConsumerClient:
    - From a connection string
    - From a shared access key
    - Creation with configuration parameters

- [send.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/send.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/send_async.py)) - Examples to send events:
    - Send event data batch
    - Send event data batch within limited size
    - Send event data batch to a specific partition determined by partition key
    - Send event data batch to a specific partition by partition id
    - Send event data batch with customized properties
    - Send events concurrently with proper thread/coroutine safety practices
      - **Note**: EventHub clients are not thread-safe or coroutine-safe

- [send_stream.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/send_stream.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/send_stream_async.py)) - Examples to do streaming sending:
    - Send in a stream

- [recv.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/recv.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/recv_async.py)) - Examples to receive events:
    - Receive events

- [recv_track_last_enqueued_event_prop.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/recv_track_last_enqueued_event_prop.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/recv_track_last_enqueued_event_prop_async.py)) - Examples to get the latest enqueued event properties of a partition while receiving:
    - Get the latest enqueued event properties of a partition while receiving

- [recv_with_custom_starting_position.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/recv_with_custom_starting_position.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/recv_with_custom_starting_position_async.py)) - Examples to start receiving from a specific position:
    - Start receiving from a specific position

- [recv_with_checkpoint_store.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/recv_with_checkpoint_store.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/recv_with_checkpoint_store_async.py)) - Examples to receive events and do checkpoint using blob checkpoint store:
    - Receive events and do checkpoint using blob checkpoint store

- [recv_with_checkpoint_by_event_count.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/recv_with_checkpoint_by_event_count.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/recv_with_checkpoint_by_event_count_async.py)) - Examples to receive events and do checkpoint by event count using blob checkpoint store:
    - Receive events and do checkpoint every fixed amount of events (e.g. checkpoint every 20 events) using blob checkpoint store

- [receive_batch_with_checkpoint.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/receive_batch_with_checkpoint.py) ([async_version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/receive_batch_with_checkpoint_async.py)) - Examples to receive events in batches and do checkpoint by the batch:
    - Receive events in batches by calling `EventHubConsumer.receive_batch` and do checkpoint with the last event of that batch.

- [recv_with_checkpoint_by_time_interval.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/recv_with_checkpoint_by_time_interval.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/recv_with_checkpoint_by_time_interval_async.py)) - Examples to receive events and do checkpoint by time interval using blob checkpoint store:
    - Receive events and do checkpoint every fixed time interval (e.g. checkpoint every 20 seconds) using blob checkpoint store

- [recv_for_period.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/recv_for_period.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/recv_for_period_async.py)) - Examples to receive events for a period of time:
    - Receive events for a period of time

- [client_identity_authentication.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/client_identity_authentication.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/client_identity_authentication_async.py)) - Examples for authentication by Azure Active Directory:
    - Authenticating and creating the client utilizing the `azure.identity` library

- [connection_string_authentication.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/connection_string_authentication.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/connection_string_authentication_async.py)) - Examples for authentication by connection string:
    - Authenticating and creating the client utilizing a connection string.

- [proxy.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/proxy.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/proxy_async.py)) - Examples to send and receive events behind a proxy:
    - Send and receive events behind a proxy

- [iot_hub_connection_string_receive_async.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/iot_hub_connection_string_receive_async.py) - Examples to receive events from an IoT Hub:
    - Convert an IoT Hub connection string to the built-in Event Hub endpoint and receive events from it

- [authenticate_with_sas_token.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/authenticate_with_sas_token.py) ([async_version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/authenticate_with_sas_token_async.py)) - Examples:
    - Utilize a SAS token to authenticate when creating an Event Hub client.

- [authenticate_with_azure_sas_credential.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/authenticate_with_azure_sas_credential.py) ([async_version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/authenticate_with_azure_sas_credential_async.py)) - Examples:
    - Utilize `azure.core.credentials.AzureSasCredential` to authenticate when creating an Event Hub client.

- [authenticate_with_azure_named_key_credential.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/authenticate_with_azure_named_key_credential.py) ([async_version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/authenticate_with_azure_named_key_credential_async.py)) - Examples:
    - Utilize `azure.core.credentials.AzureNamedKeyCredential` to authenticate when creating an Event Hub client.

- [connection_to_custom_endpoint_address.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/connection_to_custom_endpoint_address.py) ([async version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/connection_to_custom_endpoint_address_async.py)) - Examples:
    - Create EventHubProducerClient and EventHubConsumerClient that connect to a custom endpoint with a custom certificate.

- [send_and_receive_amqp_annotated_message.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/send_and_receive_amqp_annotated_message.py) ([async_version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/send_and_receive_amqp_annotated_message_async.py)) - Examples to send AMQPAnnotatedMessage to and receive events from an event hub and parse the body:
    - Send AMQPAnnotatedMessage of different body types.
    - Receive messages and parse the body according to the body type.

- [send_buffered_mode.py](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/sync_samples/send_buffered_mode.py) ([async_version](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub/samples/async_samples/send_buffered_mode_async.py)) - Examples to send events in buffered mode:
    - Send single events, which will be automatically batched.
    - Send a batch of events by enqueuing an EventDataBatch object to the buffer.
    - Send events in buffer immediately by calling `flush`.

## Prerequisites
- Python 3.8 or later.
- **Microsoft Azure Subscription:**  To use Azure services, including Azure Event Hubs, you'll need a subscription.
If you do not have an existing Azure account, you may sign up for a free trial or use your MSDN subscriber benefits when you [create an account](https://account.windowsazure.com/Home/Index).

- **Event Hubs namespace with an Event Hub:** To interact with Azure Event Hubs, you'll also need to have a namespace and Event Hub  available.
If you are not familiar with creating Azure resources, you may wish to follow the step-by-step guide
for [creating an Event Hub using the Azure portal](https://learn.microsoft.com/azure/event-hubs/event-hubs-create).
There, you can also find detailed instructions for using the Azure CLI, Azure PowerShell, or Azure Resource Manager (ARM) templates to create an Event Hub.

- **Azure Storage Account (Optional)**: To run receiving samples with blob checkpoint store for persist checkpoint, you need to [create an Azure Storage account](https://learn.microsoft.com/azure/storage/common/storage-quickstart-create-account?tabs=azure-portal) and a [Blob Container](https://learn.microsoft.com/azure/storage/blobs/storage-quickstart-blobs-portal#create-a-container).

## Setup

1. Install the Azure Event Hubs client library for Python with [pip](https://pypi.org/project/pip/):
```bash
pip install azure-eventhub
```

To run samples that utilize the Azure Active Directory for authentication, please install the `azure-identity` library:
```bash
pip install azure-identity
```

To run receiving samples that utilize blob checkpoint store for persist checkpoint, please install the corresponding checkpoint store library:
```bash
pip install azure-eventhub-checkpointstoreblob  # sync version
pip install azure-eventhub-checkpointstoreblob-aio  # async version
```
2. Clone or download this sample repository.
3. Open the sample folder in Visual Studio Code or your IDE of choice.

## Running the samples

1. Open a terminal window and `cd` to the directory that the samples are saved in.
2. Set the environment variables specified in the sample file you wish to run.
3. Follow the usage described in the file, e.g. `python send.py`.

## Next steps

Check out the [API reference documentation](https://learn.microsoft.com/python/api/overview/azure/eventhub-readme) to learn more about
what you can do with the Azure Event Hubs client library.
