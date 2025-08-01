# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import base64
import hashlib
import logging
import random
import re
import uuid
from io import SEEK_SET, UnsupportedOperation
from time import time
from typing import Any, Dict, Optional, TYPE_CHECKING
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)
from wsgiref.handlers import format_date_time

from azure.core.exceptions import AzureError, ServiceRequestError, ServiceResponseError
from azure.core.pipeline.policies import (
    BearerTokenCredentialPolicy,
    HeadersPolicy,
    HTTPPolicy,
    NetworkTraceLoggingPolicy,
    RequestHistory,
    SansIOHTTPPolicy,
)

from .authentication import AzureSigningError, StorageHttpChallenge
from .constants import DEFAULT_OAUTH_SCOPE
from .models import LocationMode, StorageErrorCode

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential
    from azure.core.pipeline.transport import (  # pylint: disable=non-abstract-transport-import
        PipelineRequest,
        PipelineResponse,
    )


_LOGGER = logging.getLogger(__name__)


def encode_base64(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    encoded = base64.b64encode(data)
    return encoded.decode("utf-8")


# Are we out of retries?
def is_exhausted(settings):
    retry_counts = (settings["total"], settings["connect"], settings["read"], settings["status"])
    retry_counts = list(filter(None, retry_counts))
    if not retry_counts:
        return False
    return min(retry_counts) < 0


def retry_hook(settings, **kwargs):
    if settings["hook"]:
        settings["hook"](retry_count=settings["count"] - 1, location_mode=settings["mode"], **kwargs)


# Is this method/status code retryable? (Based on allowlists and control
# variables such as the number of total retries to allow, whether to
# respect the Retry-After header, whether this header is present, and
# whether the returned status code is on the list of status codes to
# be retried upon on the presence of the aforementioned header)
def is_retry(response, mode):  # pylint: disable=too-many-return-statements
    status = response.http_response.status_code
    if 300 <= status < 500:
        # An exception occurred, but in most cases it was expected. Examples could
        # include a 309 Conflict or 412 Precondition Failed.
        if status == 404 and mode == LocationMode.SECONDARY:
            # Response code 404 should be retried if secondary was used.
            return True
        if status == 408:
            # Response code 408 is a timeout and should be retried.
            return True
        if status >= 400:
            error_code = response.http_response.headers.get("x-ms-copy-source-error-code")
            if error_code in [
                StorageErrorCode.OPERATION_TIMED_OUT,
                StorageErrorCode.INTERNAL_ERROR,
                StorageErrorCode.SERVER_BUSY,
            ]:
                return True
        return False
    if status >= 500:
        # Response codes above 500 with the exception of 501 Not Implemented and
        # 505 Version Not Supported indicate a server issue and should be retried.
        if status in [501, 505]:
            return False
        return True
    return False


def is_checksum_retry(response):
    # retry if invalid content md5
    if response.context.get("validate_content", False) and response.http_response.headers.get("content-md5"):
        computed_md5 = response.http_request.headers.get("content-md5", None) or encode_base64(
            StorageContentValidation.get_content_md5(response.http_response.body())
        )
        if response.http_response.headers["content-md5"] != computed_md5:
            return True
    return False


def urljoin(base_url, stub_url):
    parsed = urlparse(base_url)
    parsed = parsed._replace(path=parsed.path + "/" + stub_url)
    return parsed.geturl()


class QueueMessagePolicy(SansIOHTTPPolicy):

    def on_request(self, request):
        # Hack to fix generated code adding '/messages' after SAS parameters
        includes_messages = request.http_request.url.endswith("/messages")
        if includes_messages:
            request.http_request.url = request.http_request.url[: -(len("/messages"))]
            request.http_request.url = urljoin(request.http_request.url, "messages")

        message_id = request.context.options.pop("queue_message_id", None)
        if message_id:
            request.http_request.url = urljoin(request.http_request.url, message_id)


class StorageHeadersPolicy(HeadersPolicy):
    request_id_header_name = "x-ms-client-request-id"

    def on_request(self, request: "PipelineRequest") -> None:
        super(StorageHeadersPolicy, self).on_request(request)
        current_time = format_date_time(time())
        request.http_request.headers["x-ms-date"] = current_time

        custom_id = request.context.options.pop("client_request_id", None)
        request.http_request.headers["x-ms-client-request-id"] = custom_id or str(uuid.uuid1())

    # def on_response(self, request, response):
    #     # raise exception if the echoed client request id from the service is not identical to the one we sent
    #     if self.request_id_header_name in response.http_response.headers:

    #         client_request_id = request.http_request.headers.get(self.request_id_header_name)

    #         if response.http_response.headers[self.request_id_header_name] != client_request_id:
    #             raise AzureError(
    #                 "Echoed client request ID: {} does not match sent client request ID: {}.  "
    #                 "Service request ID: {}".format(
    #                     response.http_response.headers[self.request_id_header_name], client_request_id,
    #                     response.http_response.headers['x-ms-request-id']),
    #                 response=response.http_response
    #             )


class StorageHosts(SansIOHTTPPolicy):

    def __init__(self, hosts=None, **kwargs):  # pylint: disable=unused-argument
        self.hosts = hosts
        super(StorageHosts, self).__init__()

    def on_request(self, request: "PipelineRequest") -> None:
        request.context.options["hosts"] = self.hosts
        parsed_url = urlparse(request.http_request.url)

        # Detect what location mode we're currently requesting with
        location_mode = LocationMode.PRIMARY
        for key, value in self.hosts.items():
            if parsed_url.netloc == value:
                location_mode = key

        # See if a specific location mode has been specified, and if so, redirect
        use_location = request.context.options.pop("use_location", None)
        if use_location:
            # Lock retries to the specific location
            request.context.options["retry_to_secondary"] = False
            if use_location not in self.hosts:
                raise ValueError(f"Attempting to use undefined host location {use_location}")
            if use_location != location_mode:
                # Update request URL to use the specified location
                updated = parsed_url._replace(netloc=self.hosts[use_location])
                request.http_request.url = updated.geturl()
                location_mode = use_location

        request.context.options["location_mode"] = location_mode


class StorageLoggingPolicy(NetworkTraceLoggingPolicy):
    """A policy that logs HTTP request and response to the DEBUG logger.

    This accepts both global configuration, and per-request level with "enable_http_logger"
    """

    def __init__(self, logging_enable: bool = False, **kwargs) -> None:
        self.logging_body = kwargs.pop("logging_body", False)
        super(StorageLoggingPolicy, self).__init__(logging_enable=logging_enable, **kwargs)

    def on_request(self, request: "PipelineRequest") -> None:
        http_request = request.http_request
        options = request.context.options
        self.logging_body = self.logging_body or options.pop("logging_body", False)
        if options.pop("logging_enable", self.enable_http_logger):
            request.context["logging_enable"] = True
            if not _LOGGER.isEnabledFor(logging.DEBUG):
                return

            try:
                log_url = http_request.url
                query_params = http_request.query
                if "sig" in query_params:
                    log_url = log_url.replace(query_params["sig"], "sig=*****")
                _LOGGER.debug("Request URL: %r", log_url)
                _LOGGER.debug("Request method: %r", http_request.method)
                _LOGGER.debug("Request headers:")
                for header, value in http_request.headers.items():
                    if header.lower() == "authorization":
                        value = "*****"
                    elif header.lower() == "x-ms-copy-source" and "sig" in value:
                        # take the url apart and scrub away the signed signature
                        scheme, netloc, path, params, query, fragment = urlparse(value)
                        parsed_qs = dict(parse_qsl(query))
                        parsed_qs["sig"] = "*****"

                        # the SAS needs to be put back together
                        value = urlunparse((scheme, netloc, path, params, urlencode(parsed_qs), fragment))

                    _LOGGER.debug("    %r: %r", header, value)
                _LOGGER.debug("Request body:")

                if self.logging_body:
                    _LOGGER.debug(str(http_request.body))
                else:
                    # We don't want to log the binary data of a file upload.
                    _LOGGER.debug("Hidden body, please use logging_body to show body")
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.debug("Failed to log request: %r", err)

    def on_response(self, request: "PipelineRequest", response: "PipelineResponse") -> None:
        if response.context.pop("logging_enable", self.enable_http_logger):
            if not _LOGGER.isEnabledFor(logging.DEBUG):
                return

            try:
                _LOGGER.debug("Response status: %r", response.http_response.status_code)
                _LOGGER.debug("Response headers:")
                for res_header, value in response.http_response.headers.items():
                    _LOGGER.debug("    %r: %r", res_header, value)

                # We don't want to log binary data if the response is a file.
                _LOGGER.debug("Response content:")
                pattern = re.compile(r'attachment; ?filename=["\w.]+', re.IGNORECASE)
                header = response.http_response.headers.get("content-disposition")
                resp_content_type = response.http_response.headers.get("content-type", "")

                if header and pattern.match(header):
                    filename = header.partition("=")[2]
                    _LOGGER.debug("File attachments: %s", filename)
                elif resp_content_type.endswith("octet-stream"):
                    _LOGGER.debug("Body contains binary data.")
                elif resp_content_type.startswith("image"):
                    _LOGGER.debug("Body contains image data.")

                if self.logging_body and resp_content_type.startswith("text"):
                    _LOGGER.debug(response.http_response.text())
                elif self.logging_body:
                    try:
                        _LOGGER.debug(response.http_response.body())
                    except ValueError:
                        _LOGGER.debug("Body is streamable")

            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.debug("Failed to log response: %s", repr(err))


class StorageRequestHook(SansIOHTTPPolicy):

    def __init__(self, **kwargs):
        self._request_callback = kwargs.get("raw_request_hook")
        super(StorageRequestHook, self).__init__()

    def on_request(self, request: "PipelineRequest") -> None:
        request_callback = request.context.options.pop("raw_request_hook", self._request_callback)
        if request_callback:
            request_callback(request)


class StorageResponseHook(HTTPPolicy):

    def __init__(self, **kwargs):
        self._response_callback = kwargs.get("raw_response_hook")
        super(StorageResponseHook, self).__init__()

    def send(self, request: "PipelineRequest") -> "PipelineResponse":
        # Values could be 0
        data_stream_total = request.context.get("data_stream_total")
        if data_stream_total is None:
            data_stream_total = request.context.options.pop("data_stream_total", None)
        download_stream_current = request.context.get("download_stream_current")
        if download_stream_current is None:
            download_stream_current = request.context.options.pop("download_stream_current", None)
        upload_stream_current = request.context.get("upload_stream_current")
        if upload_stream_current is None:
            upload_stream_current = request.context.options.pop("upload_stream_current", None)

        response_callback = request.context.get("response_callback") or request.context.options.pop(
            "raw_response_hook", self._response_callback
        )

        response = self.next.send(request)

        will_retry = is_retry(response, request.context.options.get("mode")) or is_checksum_retry(response)
        # Auth error could come from Bearer challenge, in which case this request will be made again
        is_auth_error = response.http_response.status_code == 401
        should_update_counts = not (will_retry or is_auth_error)

        if should_update_counts and download_stream_current is not None:
            download_stream_current += int(response.http_response.headers.get("Content-Length", 0))
            if data_stream_total is None:
                content_range = response.http_response.headers.get("Content-Range")
                if content_range:
                    data_stream_total = int(content_range.split(" ", 1)[1].split("/", 1)[1])
                else:
                    data_stream_total = download_stream_current
        elif should_update_counts and upload_stream_current is not None:
            upload_stream_current += int(response.http_request.headers.get("Content-Length", 0))
        for pipeline_obj in [request, response]:
            if hasattr(pipeline_obj, "context"):
                pipeline_obj.context["data_stream_total"] = data_stream_total
                pipeline_obj.context["download_stream_current"] = download_stream_current
                pipeline_obj.context["upload_stream_current"] = upload_stream_current
        if response_callback:
            response_callback(response)
            request.context["response_callback"] = response_callback
        return response


class StorageContentValidation(SansIOHTTPPolicy):
    """A simple policy that sends the given headers
    with the request.

    This will overwrite any headers already defined in the request.
    """

    header_name = "Content-MD5"

    def __init__(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        super(StorageContentValidation, self).__init__()

    @staticmethod
    def get_content_md5(data):
        # Since HTTP does not differentiate between no content and empty content,
        # we have to perform a None check.
        data = data or b""
        md5 = hashlib.md5()  # nosec
        if isinstance(data, bytes):
            md5.update(data)
        elif hasattr(data, "read"):
            pos = 0
            try:
                pos = data.tell()
            except:  # pylint: disable=bare-except
                pass
            for chunk in iter(lambda: data.read(4096), b""):
                md5.update(chunk)
            try:
                data.seek(pos, SEEK_SET)
            except (AttributeError, IOError) as exc:
                raise ValueError("Data should be bytes or a seekable file-like object.") from exc
        else:
            raise ValueError("Data should be bytes or a seekable file-like object.")

        return md5.digest()

    def on_request(self, request: "PipelineRequest") -> None:
        validate_content = request.context.options.pop("validate_content", False)
        if validate_content and request.http_request.method != "GET":
            computed_md5 = encode_base64(StorageContentValidation.get_content_md5(request.http_request.data))
            request.http_request.headers[self.header_name] = computed_md5
            request.context["validate_content_md5"] = computed_md5
        request.context["validate_content"] = validate_content

    def on_response(self, request: "PipelineRequest", response: "PipelineResponse") -> None:
        if response.context.get("validate_content", False) and response.http_response.headers.get("content-md5"):
            computed_md5 = request.context.get("validate_content_md5") or encode_base64(
                StorageContentValidation.get_content_md5(response.http_response.body())
            )
            if response.http_response.headers["content-md5"] != computed_md5:
                raise AzureError(
                    (
                        f"MD5 mismatch. Expected value is '{response.http_response.headers['content-md5']}', "
                        f"computed value is '{computed_md5}'."
                    ),
                    response=response.http_response,
                )


class StorageRetryPolicy(HTTPPolicy):
    """
    The base class for Exponential and Linear retries containing shared code.
    """

    total_retries: int
    """The max number of retries."""
    connect_retries: int
    """The max number of connect retries."""
    retry_read: int
    """The max number of read retries."""
    retry_status: int
    """The max number of status retries."""
    retry_to_secondary: bool
    """Whether the secondary endpoint should be retried."""

    def __init__(self, **kwargs: Any) -> None:
        self.total_retries = kwargs.pop("retry_total", 10)
        self.connect_retries = kwargs.pop("retry_connect", 3)
        self.read_retries = kwargs.pop("retry_read", 3)
        self.status_retries = kwargs.pop("retry_status", 3)
        self.retry_to_secondary = kwargs.pop("retry_to_secondary", False)
        super(StorageRetryPolicy, self).__init__()

    def _set_next_host_location(self, settings: Dict[str, Any], request: "PipelineRequest") -> None:
        """
        A function which sets the next host location on the request, if applicable.

        :param Dict[str, Any] settings: The configurable values pertaining to the next host location.
        :param PipelineRequest request: A pipeline request object.
        """
        if settings["hosts"] and all(settings["hosts"].values()):
            url = urlparse(request.url)
            # If there's more than one possible location, retry to the alternative
            if settings["mode"] == LocationMode.PRIMARY:
                settings["mode"] = LocationMode.SECONDARY
            else:
                settings["mode"] = LocationMode.PRIMARY
            updated = url._replace(netloc=settings["hosts"].get(settings["mode"]))
            request.url = updated.geturl()

    def configure_retries(self, request: "PipelineRequest") -> Dict[str, Any]:
        """
        Configure the retry settings for the request.

        :param request: A pipeline request object.
        :type request: ~azure.core.pipeline.PipelineRequest
        :return: A dictionary containing the retry settings.
        :rtype: Dict[str, Any]
        """
        body_position = None
        if hasattr(request.http_request.body, "read"):
            try:
                body_position = request.http_request.body.tell()
            except (AttributeError, UnsupportedOperation):
                # if body position cannot be obtained, then retries will not work
                pass
        options = request.context.options
        return {
            "total": options.pop("retry_total", self.total_retries),
            "connect": options.pop("retry_connect", self.connect_retries),
            "read": options.pop("retry_read", self.read_retries),
            "status": options.pop("retry_status", self.status_retries),
            "retry_secondary": options.pop("retry_to_secondary", self.retry_to_secondary),
            "mode": options.pop("location_mode", LocationMode.PRIMARY),
            "hosts": options.pop("hosts", None),
            "hook": options.pop("retry_hook", None),
            "body_position": body_position,
            "count": 0,
            "history": [],
        }

    def get_backoff_time(self, settings: Dict[str, Any]) -> float:  # pylint: disable=unused-argument
        """Formula for computing the current backoff.
        Should be calculated by child class.

        :param Dict[str, Any] settings: The configurable values pertaining to the backoff time.
        :return: The backoff time.
        :rtype: float
        """
        return 0

    def sleep(self, settings, transport):
        """Sleep for the backoff time.

        :param Dict[str, Any] settings: The configurable values pertaining to the sleep operation.
        :param transport: The transport to use for sleeping.
        :type transport:
            ~azure.core.pipeline.transport.AsyncioBaseTransport or
            ~azure.core.pipeline.transport.BaseTransport
        """
        backoff = self.get_backoff_time(settings)
        if not backoff or backoff < 0:
            return
        transport.sleep(backoff)

    def increment(
        self,
        settings: Dict[str, Any],
        request: "PipelineRequest",
        response: Optional["PipelineResponse"] = None,
        error: Optional[AzureError] = None,
    ) -> bool:
        """Increment the retry counters.

        :param Dict[str, Any] settings: The configurable values pertaining to the increment operation.
        :param request: A pipeline request object.
        :type request: ~azure.core.pipeline.PipelineRequest
        :param response: A pipeline response object.
        :type response: ~azure.core.pipeline.PipelineResponse or None
        :param error: An error encountered during the request, or
            None if the response was received successfully.
        :type error: ~azure.core.exceptions.AzureError or None
        :return: Whether the retry attempts are exhausted.
        :rtype: bool
        """
        settings["total"] -= 1

        if error and isinstance(error, ServiceRequestError):
            # Errors when we're fairly sure that the server did not receive the
            # request, so it should be safe to retry.
            settings["connect"] -= 1
            settings["history"].append(RequestHistory(request, error=error))

        elif error and isinstance(error, ServiceResponseError):
            # Errors that occur after the request has been started, so we should
            # assume that the server began processing it.
            settings["read"] -= 1
            settings["history"].append(RequestHistory(request, error=error))

        else:
            # Incrementing because of a server error like a 500 in
            # status_forcelist and a the given method is in the allowlist
            if response:
                settings["status"] -= 1
                settings["history"].append(RequestHistory(request, http_response=response))

        if not is_exhausted(settings):
            if request.method not in ["PUT"] and settings["retry_secondary"]:
                self._set_next_host_location(settings, request)

            # rewind the request body if it is a stream
            if request.body and hasattr(request.body, "read"):
                # no position was saved, then retry would not work
                if settings["body_position"] is None:
                    return False
                try:
                    # attempt to rewind the body to the initial position
                    request.body.seek(settings["body_position"], SEEK_SET)
                except (UnsupportedOperation, ValueError):
                    # if body is not seekable, then retry would not work
                    return False
            settings["count"] += 1
            return True
        return False

    def send(self, request):
        """Send the request with retry logic.

        :param request: A pipeline request object.
        :type request: ~azure.core.pipeline.PipelineRequest
        :return: A pipeline response object.
        :rtype: ~azure.core.pipeline.PipelineResponse
        """
        retries_remaining = True
        response = None
        retry_settings = self.configure_retries(request)
        while retries_remaining:
            try:
                response = self.next.send(request)
                if is_retry(response, retry_settings["mode"]) or is_checksum_retry(response):
                    retries_remaining = self.increment(
                        retry_settings, request=request.http_request, response=response.http_response
                    )
                    if retries_remaining:
                        retry_hook(
                            retry_settings, request=request.http_request, response=response.http_response, error=None
                        )
                        self.sleep(retry_settings, request.context.transport)
                        continue
                break
            except AzureError as err:
                if isinstance(err, AzureSigningError):
                    raise
                retries_remaining = self.increment(retry_settings, request=request.http_request, error=err)
                if retries_remaining:
                    retry_hook(retry_settings, request=request.http_request, response=None, error=err)
                    self.sleep(retry_settings, request.context.transport)
                    continue
                raise err
        if retry_settings["history"]:
            response.context["history"] = retry_settings["history"]
        response.http_response.location_mode = retry_settings["mode"]
        return response


class ExponentialRetry(StorageRetryPolicy):
    """Exponential retry."""

    initial_backoff: int
    """The initial backoff interval, in seconds, for the first retry."""
    increment_base: int
    """The base, in seconds, to increment the initial_backoff by after the
    first retry."""
    random_jitter_range: int
    """A number in seconds which indicates a range to jitter/randomize for the back-off interval."""

    def __init__(
        self,
        initial_backoff: int = 15,
        increment_base: int = 3,
        retry_total: int = 3,
        retry_to_secondary: bool = False,
        random_jitter_range: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Constructs an Exponential retry object. The initial_backoff is used for
        the first retry. Subsequent retries are retried after initial_backoff +
        increment_power^retry_count seconds.

        :param int initial_backoff:
            The initial backoff interval, in seconds, for the first retry.
        :param int increment_base:
            The base, in seconds, to increment the initial_backoff by after the
            first retry.
        :param int retry_total:
            The maximum number of retry attempts.
        :param bool retry_to_secondary:
            Whether the request should be retried to secondary, if able. This should
            only be enabled of RA-GRS accounts are used and potentially stale data
            can be handled.
        :param int random_jitter_range:
            A number in seconds which indicates a range to jitter/randomize for the back-off interval.
            For example, a random_jitter_range of 3 results in the back-off interval x to vary between x+3 and x-3.
        """
        self.initial_backoff = initial_backoff
        self.increment_base = increment_base
        self.random_jitter_range = random_jitter_range
        super(ExponentialRetry, self).__init__(retry_total=retry_total, retry_to_secondary=retry_to_secondary, **kwargs)

    def get_backoff_time(self, settings: Dict[str, Any]) -> float:
        """
        Calculates how long to sleep before retrying.

        :param Dict[str, Any] settings: The configurable values pertaining to get backoff time.
        :return:
            A float indicating how long to wait before retrying the request,
            or None to indicate no retry should be performed.
        :rtype: float
        """
        random_generator = random.Random()
        backoff = self.initial_backoff + (0 if settings["count"] == 0 else pow(self.increment_base, settings["count"]))
        random_range_start = backoff - self.random_jitter_range if backoff > self.random_jitter_range else 0
        random_range_end = backoff + self.random_jitter_range
        return random_generator.uniform(random_range_start, random_range_end)


class LinearRetry(StorageRetryPolicy):
    """Linear retry."""

    initial_backoff: int
    """The backoff interval, in seconds, between retries."""
    random_jitter_range: int
    """A number in seconds which indicates a range to jitter/randomize for the back-off interval."""

    def __init__(
        self,
        backoff: int = 15,
        retry_total: int = 3,
        retry_to_secondary: bool = False,
        random_jitter_range: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Constructs a Linear retry object.

        :param int backoff:
            The backoff interval, in seconds, between retries.
        :param int retry_total:
            The maximum number of retry attempts.
        :param bool retry_to_secondary:
            Whether the request should be retried to secondary, if able. This should
            only be enabled of RA-GRS accounts are used and potentially stale data
            can be handled.
        :param int random_jitter_range:
            A number in seconds which indicates a range to jitter/randomize for the back-off interval.
            For example, a random_jitter_range of 3 results in the back-off interval x to vary between x+3 and x-3.
        """
        self.backoff = backoff
        self.random_jitter_range = random_jitter_range
        super(LinearRetry, self).__init__(retry_total=retry_total, retry_to_secondary=retry_to_secondary, **kwargs)

    def get_backoff_time(self, settings: Dict[str, Any]) -> float:
        """
        Calculates how long to sleep before retrying.

        :param Dict[str, Any] settings: The configurable values pertaining to the backoff time.
        :return:
            A float indicating how long to wait before retrying the request,
            or None to indicate no retry should be performed.
        :rtype: float
        """
        random_generator = random.Random()
        # the backoff interval normally does not change, however there is the possibility
        # that it was modified by accessing the property directly after initializing the object
        random_range_start = self.backoff - self.random_jitter_range if self.backoff > self.random_jitter_range else 0
        random_range_end = self.backoff + self.random_jitter_range
        return random_generator.uniform(random_range_start, random_range_end)


class StorageBearerTokenCredentialPolicy(BearerTokenCredentialPolicy):
    """Custom Bearer token credential policy for following Storage Bearer challenges"""

    def __init__(self, credential: "TokenCredential", audience: str, **kwargs: Any) -> None:
        super(StorageBearerTokenCredentialPolicy, self).__init__(credential, audience, **kwargs)

    def on_challenge(self, request: "PipelineRequest", response: "PipelineResponse") -> bool:
        """Handle the challenge from the service and authorize the request.

        :param request: The request object.
        :type request: ~azure.core.pipeline.PipelineRequest
        :param response: The response object.
        :type response: ~azure.core.pipeline.PipelineResponse
        :return: True if the request was authorized, False otherwise.
        :rtype: bool
        """
        try:
            auth_header = response.http_response.headers.get("WWW-Authenticate")
            challenge = StorageHttpChallenge(auth_header)
        except ValueError:
            return False

        scope = challenge.resource_id + DEFAULT_OAUTH_SCOPE
        self.authorize_request(request, scope, tenant_id=challenge.tenant_id)

        return True
