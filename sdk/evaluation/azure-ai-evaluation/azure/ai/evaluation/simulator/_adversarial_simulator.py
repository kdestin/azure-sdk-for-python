# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# noqa: E501
# pylint: disable=E0401,E0611
import asyncio
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Union, cast
import uuid

from tqdm import tqdm

from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._common.utils import validate_azure_ai_project, is_onedp_project
from azure.ai.evaluation._common.onedp._client import AIProjectClient
from azure.ai.evaluation._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException
from azure.ai.evaluation._http_utils import get_async_http_client
from azure.ai.evaluation._model_configurations import AzureAIProject
from azure.ai.evaluation.simulator import AdversarialScenario, AdversarialScenarioJailbreak
from azure.ai.evaluation.simulator._adversarial_scenario import _UnstableAdversarialScenario
from azure.ai.evaluation._constants import TokenScope
from azure.core.credentials import TokenCredential
from azure.core.pipeline.policies import AsyncRetryPolicy, RetryMode

from ._constants import SupportedLanguages
from ._conversation import (
    CallbackConversationBot,
    MultiModalConversationBot,
    ConversationBot,
    ConversationRole,
    ConversationTurn,
)
from ._conversation._conversation import simulate_conversation
from ._model_tools import (
    AdversarialTemplateHandler,
    ManagedIdentityAPITokenManager,
    ProxyChatCompletionsModel,
    RAIClient,
)
from ._model_tools._template_handler import AdversarialTemplate, TemplateParameters
from ._utils import JsonLineList

logger = logging.getLogger(__name__)


@experimental
class AdversarialSimulator:
    """
    Initializes the adversarial simulator with a project scope.

    :param azure_ai_project: The Azure AI project, which can either be a string representing the project endpoint
        or an instance of AzureAIProject. It contains subscription id, resource group, and project name.
    :type azure_ai_project: Union[str, AzureAIProject]
    :param credential: The credential for connecting to Azure AI project.
    :type credential: ~azure.core.credentials.TokenCredential

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_simulate.py
            :start-after: [START adversarial_scenario]
            :end-before: [END adversarial_scenario]
            :language: python
            :dedent: 8
            :caption: Run the AdversarialSimulator with an AdversarialConversation scenario to produce 2 results with
                2 conversation turns each (4 messages per result).
    """

    def __init__(self, *, azure_ai_project: Union[str, AzureAIProject], credential: TokenCredential):
        """Constructor."""

        if is_onedp_project(azure_ai_project):
            self.azure_ai_project = azure_ai_project
            self.credential = cast(TokenCredential, credential)
            self.token_manager = ManagedIdentityAPITokenManager(
                token_scope=TokenScope.COGNITIVE_SERVICES_MANAGEMENT,
                logger=logging.getLogger("AdversarialSimulator"),
                credential=self.credential,
            )
            self.rai_client = AIProjectClient(endpoint=azure_ai_project, credential=credential)
        else:
            try:
                self.azure_ai_project = validate_azure_ai_project(azure_ai_project)
            except EvaluationException as e:
                raise EvaluationException(
                    message=e.message,
                    internal_message=e.internal_message,
                    target=ErrorTarget.ADVERSARIAL_SIMULATOR,
                    category=e.category,
                    blame=e.blame,
                ) from e
            self.credential = cast(TokenCredential, credential)
            self.token_manager = ManagedIdentityAPITokenManager(
                token_scope=TokenScope.DEFAULT_AZURE_MANAGEMENT,
                logger=logging.getLogger("AdversarialSimulator"),
                credential=self.credential,
            )
            self.rai_client = RAIClient(azure_ai_project=self.azure_ai_project, token_manager=self.token_manager)

        self.adversarial_template_handler = AdversarialTemplateHandler(
            azure_ai_project=self.azure_ai_project, rai_client=self.rai_client
        )

    def _ensure_service_dependencies(self):
        if self.rai_client is None:
            msg = "RAI service is required for simulation, but an RAI client was not provided."
            raise EvaluationException(
                message=msg,
                internal_message=msg,
                target=ErrorTarget.ADVERSARIAL_SIMULATOR,
                category=ErrorCategory.MISSING_FIELD,
                blame=ErrorBlame.USER_ERROR,
            )

    # pylint: disable=too-many-locals
    async def __call__(
        self,
        *,
        # Note: the scenario input also accepts inputs from _PrivateAdversarialScenario, but that's
        # not stated since those values are nominally for internal use only.
        scenario: AdversarialScenario,
        target: Callable,
        max_conversation_turns: int = 1,
        max_simulation_results: int = 3,
        api_call_retry_limit: int = 3,
        api_call_retry_sleep_sec: int = 1,
        api_call_delay_sec: int = 0,
        concurrent_async_task: int = 3,
        language: SupportedLanguages = SupportedLanguages.English,
        randomize_order: bool = True,
        randomization_seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Executes the adversarial simulation against a specified target function asynchronously.

        :keyword scenario: Enum value specifying the adversarial scenario used for generating inputs.
         example:

         - :py:const:`azure.ai.evaluation.simulator.AdversarialScenario.ADVERSARIAL_QA`
         - :py:const:`azure.ai.evaluation.simulator.AdversarialScenario.ADVERSARIAL_CONVERSATION`
        :paramtype scenario: azure.ai.evaluation.simulator.AdversarialScenario
        :keyword target: The target function to simulate adversarial inputs against.
            This function should be asynchronous and accept a dictionary representing the adversarial input.
        :paramtype target: Callable
        :keyword max_conversation_turns: The maximum number of conversation turns to simulate.
            Defaults to 1.
        :paramtype max_conversation_turns: int
        :keyword max_simulation_results: The maximum number of simulation results to return.
            Defaults to 3.
        :paramtype max_simulation_results: int
        :keyword api_call_retry_limit: The maximum number of retries for each API call within the simulation.
            Defaults to 3.
        :paramtype api_call_retry_limit: int
        :keyword api_call_retry_sleep_sec: The sleep duration (in seconds) between retries for API calls.
            Defaults to 1 second.
        :paramtype api_call_retry_sleep_sec: int
        :keyword api_call_delay_sec: The delay (in seconds) before making an API call.
            This can be used to avoid hitting rate limits. Defaults to 0 seconds.
        :paramtype api_call_delay_sec: int
        :keyword concurrent_async_task: The number of asynchronous tasks to run concurrently during the simulation.
            Defaults to 3.
        :paramtype concurrent_async_task: int
        :keyword language: The language in which the conversation should be generated. Defaults to English.
        :paramtype language: azure.ai.evaluation.simulator.SupportedLanguages
        :keyword randomize_order: Whether or not the order of the prompts should be randomized. Defaults to True.
        :paramtype randomize_order: bool
        :keyword randomization_seed: The seed used to randomize prompt selection. If unset, the system's
            default seed is used. Defaults to None.
        :paramtype randomization_seed: Optional[int]
        :return: A list of dictionaries, each representing a simulated conversation. Each dictionary contains:

         - 'template_parameters': A dictionary with parameters used in the conversation template,
            including 'conversation_starter'.
         - 'messages': A list of dictionaries, each representing a turn in the conversation.
            Each message dictionary includes 'content' (the message text) and
            'role' (indicating whether the message is from the 'user' or the 'assistant').
         - '**$schema**': A string indicating the schema URL for the conversation format.

         The 'content' for 'assistant' role messages may includes the messages that your callback returned.
        :rtype: List[Dict[str, Any]]
        """

        # validate the inputs
        if scenario != AdversarialScenario.ADVERSARIAL_CONVERSATION:
            max_conversation_turns = 2
        else:
            max_conversation_turns = max_conversation_turns * 2
        if not (
            scenario in AdversarialScenario.__members__.values()
            or scenario in _UnstableAdversarialScenario.__members__.values()
        ):
            msg = f"Invalid scenario: {scenario}. Supported scenarios are: {AdversarialScenario.__members__.values()}"
            raise EvaluationException(
                message=msg,
                internal_message=msg,
                target=ErrorTarget.ADVERSARIAL_SIMULATOR,
                category=ErrorCategory.INVALID_VALUE,
                blame=ErrorBlame.USER_ERROR,
            )
        self._ensure_service_dependencies()
        templates = await self.adversarial_template_handler._get_content_harm_template_collections(scenario.value)
        if len(templates) == 0:
            raise EvaluationException(
                message="Templates not found. Please check https://aka.ms/azureaiadvsimulator-regionsupport for region support.",
                internal_message="Please check https://aka.ms/azureaiadvsimulator-regionsupport for region support.",
                target=ErrorTarget.ADVERSARIAL_SIMULATOR,
            )
        simulation_id = str(uuid.uuid4())
        logger.warning("Use simulation_id to help debug the issue: %s", str(simulation_id))
        concurrent_async_task = min(concurrent_async_task, 1000)
        semaphore = asyncio.Semaphore(concurrent_async_task)
        sim_results = []
        tasks = []
        total_tasks = sum(len(t.template_parameters) for t in templates)
        if max_simulation_results > total_tasks:
            logger.warning(
                "Cannot provide %s results due to maximum number of adversarial simulations that can be generated: %s."
                "\n %s simulations will be generated.",
                max_simulation_results,
                total_tasks,
                total_tasks,
            )
        total_tasks = min(total_tasks, max_simulation_results)
        _jailbreak_type = kwargs.get("_jailbreak_type", None)
        if _jailbreak_type:
            if isinstance(self.rai_client, RAIClient):
                jailbreak_dataset = await self.rai_client.get_jailbreaks_dataset(type=_jailbreak_type)
            elif isinstance(self.rai_client, AIProjectClient):
                jailbreak_dataset = self.rai_client.red_teams.get_jail_break_dataset_with_type(type=_jailbreak_type)
        progress_bar = tqdm(
            total=total_tasks,
            desc="generating jailbreak simulations" if _jailbreak_type else "generating simulations",
            ncols=100,
            unit="simulations",
        )
        if randomize_order:
            # The template parameter lists are persistent across sim runs within a session,
            # So randomize a the selection instead of the parameter list directly,
            # or a potentially large deep copy.
            if randomization_seed is not None:
                # Create a local random instance to avoid polluting global state
                local_random = random.Random(randomization_seed)
                local_random.shuffle(templates)
            else:
                random.shuffle(templates)

        # Prepare task parameters based on scenario - but use a single append call for all scenarios
        tasks = []
        template_parameter_pairs = []

        if scenario == AdversarialScenario.ADVERSARIAL_CONVERSATION:
            # For ADVERSARIAL_CONVERSATION, flatten the parameters
            for i, template in enumerate(templates):
                if not template.template_parameters:
                    continue
                for parameter in template.template_parameters:
                    template_parameter_pairs.append((template, parameter))
        else:
            # Use original logic for other scenarios - zip parameters
            parameter_lists = [t.template_parameters for t in templates]
            zipped_parameters = list(zip(*parameter_lists))

            for param_group in zipped_parameters:
                for template, parameter in zip(templates, param_group):
                    template_parameter_pairs.append((template, parameter))

        # Limit to max_simulation_results if needed
        if len(template_parameter_pairs) > max_simulation_results:
            template_parameter_pairs = template_parameter_pairs[
                :max_simulation_results
            ]  # Create a seeded random instance for jailbreak selection if randomization_seed is provided
        jailbreak_random = None
        if _jailbreak_type == "upia" and randomization_seed is not None:
            jailbreak_random = random.Random(randomization_seed)

        # Single task append loop for all scenarios
        for template, parameter in template_parameter_pairs:
            if _jailbreak_type == "upia":
                if jailbreak_random is not None:
                    selected_jailbreak = jailbreak_random.choice(jailbreak_dataset)
                else:
                    selected_jailbreak = random.choice(jailbreak_dataset)
                parameter = self._add_jailbreak_parameter(parameter, selected_jailbreak)

            tasks.append(
                asyncio.create_task(
                    self._simulate_async(
                        target=target,
                        template=template,
                        parameters=parameter,
                        max_conversation_turns=max_conversation_turns,
                        api_call_retry_limit=api_call_retry_limit,
                        api_call_retry_sleep_sec=api_call_retry_sleep_sec,
                        api_call_delay_sec=api_call_delay_sec,
                        language=language,
                        semaphore=semaphore,
                        scenario=scenario,
                        simulation_id=simulation_id,
                    )
                )
            )

        for task in asyncio.as_completed(tasks):
            sim_results.append(await task)
            progress_bar.update(1)
        progress_bar.close()

        return JsonLineList(sim_results)

    def _to_chat_protocol(
        self,
        *,
        conversation_history: List[ConversationTurn],
        template_parameters: Optional[Dict[str, Union[str, Dict[str, str]]]] = None,
    ):
        if template_parameters is None:
            template_parameters = {}
        messages = []
        for _, m in enumerate(conversation_history):
            message = {"content": m.message, "role": m.role.value}
            if m.full_response is not None and "context" in m.full_response:
                message["context"] = m.full_response["context"]
            messages.append(message)
        conversation_category = cast(Dict[str, str], template_parameters.pop("metadata", {})).get("Category")
        template_parameters["metadata"] = {}
        for key in (
            "conversation_starter",
            "group_of_people",
            "target_population",
            "topic",
            "ch_template_placeholder",
            "chatbot_name",
            "name",
            "group",
        ):
            template_parameters.pop(key, None)
        if conversation_category:
            template_parameters["category"] = conversation_category
        return {
            "template_parameters": template_parameters,
            "messages": messages,
            "$schema": "http://azureml/sdk-2-0/ChatConversation.json",
        }

    async def _simulate_async(
        self,
        *,
        target: Callable,
        template: AdversarialTemplate,
        parameters: TemplateParameters,
        max_conversation_turns: int,
        api_call_retry_limit: int,
        api_call_retry_sleep_sec: int,
        api_call_delay_sec: int,
        language: SupportedLanguages,
        semaphore: asyncio.Semaphore,
        scenario: Union[AdversarialScenario, AdversarialScenarioJailbreak],
        simulation_id: str = "",
    ) -> List[Dict]:
        user_bot = self._setup_bot(
            role=ConversationRole.USER,
            template=template,
            parameters=parameters,
            scenario=scenario,
            simulation_id=simulation_id,
        )
        system_bot = self._setup_bot(
            target=target, role=ConversationRole.ASSISTANT, template=template, parameters=parameters, scenario=scenario
        )
        bots = [user_bot, system_bot]

        async def run_simulation(session_obj):
            async with semaphore:
                _, conversation_history = await simulate_conversation(
                    bots=bots,
                    session=session_obj,
                    turn_limit=max_conversation_turns,
                    api_call_delay_sec=api_call_delay_sec,
                    language=language,
                )
            return conversation_history

        if isinstance(self.rai_client, AIProjectClient):
            session = self.rai_client
        else:
            session = get_async_http_client().with_policies(
                retry_policy=AsyncRetryPolicy(
                    retry_total=api_call_retry_limit,
                    retry_backoff_factor=api_call_retry_sleep_sec,
                    retry_mode=RetryMode.Fixed,
                )
            )
        conversation_history = await run_simulation(session)

        return self._to_chat_protocol(
            conversation_history=conversation_history,
            template_parameters=cast(Dict[str, Union[str, Dict[str, str]]], parameters),
        )

    def _get_user_proxy_completion_model(
        self, template_key: str, template_parameters: TemplateParameters, simulation_id: str = ""
    ) -> ProxyChatCompletionsModel:
        endpoint_url = (
            self.rai_client._config.endpoint + "/redTeams/simulation/chat/completions/submit"
            if isinstance(self.rai_client, AIProjectClient)
            else self.rai_client.simulation_submit_endpoint
        )
        return ProxyChatCompletionsModel(
            name="raisvc_proxy_model",
            template_key=template_key,
            template_parameters=template_parameters,
            endpoint_url=endpoint_url,
            token_manager=self.token_manager,
            api_version="2023-07-01-preview",
            max_tokens=1200,
            temperature=0.0,
            simulation_id=simulation_id,
        )

    def _setup_bot(
        self,
        *,
        role: ConversationRole,
        template: AdversarialTemplate,
        parameters: TemplateParameters,
        target: Optional[Callable] = None,
        scenario: Union[AdversarialScenario, AdversarialScenarioJailbreak],
        simulation_id: str = "",
    ) -> ConversationBot:
        if role is ConversationRole.USER:
            model = self._get_user_proxy_completion_model(
                template_key=template.template_name,
                template_parameters=parameters,
                simulation_id=simulation_id,
            )
            return ConversationBot(
                role=role,
                model=model,
                conversation_template=str(template),
                instantiation_parameters=parameters,
            )

        if role is ConversationRole.ASSISTANT:
            if target is None:
                msg = "Cannot setup system bot. Target is None"

                raise EvaluationException(
                    message=msg,
                    internal_message=msg,
                    target=ErrorTarget.ADVERSARIAL_SIMULATOR,
                    error_category=ErrorCategory.INVALID_VALUE,
                    blame=ErrorBlame.SYSTEM_ERROR,
                )

            class DummyModel:
                def __init__(self):
                    self.name = "dummy_model"

                def __call__(self) -> None:
                    pass

            if scenario in [
                _UnstableAdversarialScenario.ADVERSARIAL_IMAGE_GEN,
                _UnstableAdversarialScenario.ADVERSARIAL_IMAGE_MULTIMODAL,
            ]:
                return MultiModalConversationBot(
                    callback=target,
                    role=role,
                    model=DummyModel(),
                    user_template=str(template),
                    user_template_parameters=parameters,
                    rai_client=self.rai_client,
                    conversation_template="",
                    instantiation_parameters={},
                )

            return CallbackConversationBot(
                callback=target,
                role=role,
                model=DummyModel(),
                user_template=str(template),
                user_template_parameters=parameters,
                conversation_template="",
                instantiation_parameters={},
            )

        msg = "Invalid value for enum ConversationRole. This should never happen."
        raise EvaluationException(
            message=msg,
            internal_message=msg,
            target=ErrorTarget.ADVERSARIAL_SIMULATOR,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.SYSTEM_ERROR,
        )

    def _add_jailbreak_parameter(self, parameters: TemplateParameters, to_join: str) -> TemplateParameters:
        parameters["jailbreak_string"] = to_join
        return parameters

    def call_sync(
        self,
        *,
        scenario: AdversarialScenario,
        max_conversation_turns: int,
        max_simulation_results: int,
        target: Callable,
        api_call_retry_limit: int,
        api_call_retry_sleep_sec: int,
        api_call_delay_sec: int,
        concurrent_async_task: int,
    ) -> List[Dict[str, Any]]:
        """Call the adversarial simulator synchronously.
        :keyword scenario: Enum value specifying the adversarial scenario used for generating inputs.
        example:

         - :py:const:`azure.ai.evaluation.simulator.adversarial_scenario.AdversarialScenario.ADVERSARIAL_QA`
         - :py:const:`azure.ai.evaluation.simulator.adversarial_scenario.AdversarialScenario.ADVERSARIAL_CONVERSATION`
        :paramtype scenario: azure.ai.evaluation.simulator.adversarial_scenario.AdversarialScenario

        :keyword max_conversation_turns: The maximum number of conversation turns to simulate.
        :paramtype max_conversation_turns: int
        :keyword max_simulation_results: The maximum number of simulation results to return.
        :paramtype max_simulation_results: int
        :keyword target: The target function to simulate adversarial inputs against.
        :paramtype target: Callable
        :keyword api_call_retry_limit: The maximum number of retries for each API call within the simulation.
        :paramtype api_call_retry_limit: int
        :keyword api_call_retry_sleep_sec: The sleep duration (in seconds) between retries for API calls.
        :paramtype api_call_retry_sleep_sec: int
        :keyword api_call_delay_sec: The delay (in seconds) before making an API call.
        :paramtype api_call_delay_sec: int
        :keyword concurrent_async_task: The number of asynchronous tasks to run concurrently during the simulation.
        :paramtype concurrent_async_task: int
        :return: A list of dictionaries, each representing a simulated conversation.
        :rtype: List[Dict[str, Any]]
        """
        # Running the async method in a synchronous context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If the loop is already running, use run_until_complete
            # Note: This approach might not be suitable in all contexts, especially with nested async calls
            future = asyncio.ensure_future(
                self(
                    scenario=scenario,
                    max_conversation_turns=max_conversation_turns,
                    max_simulation_results=max_simulation_results,
                    target=target,
                    api_call_retry_limit=api_call_retry_limit,
                    api_call_retry_sleep_sec=api_call_retry_sleep_sec,
                    api_call_delay_sec=api_call_delay_sec,
                    concurrent_async_task=concurrent_async_task,
                )
            )
            return loop.run_until_complete(future)

        # If no event loop is running, use asyncio.run (Python 3.7+)
        return asyncio.run(
            self(
                scenario=scenario,
                max_conversation_turns=max_conversation_turns,
                max_simulation_results=max_simulation_results,
                target=target,
                api_call_retry_limit=api_call_retry_limit,
                api_call_retry_sleep_sec=api_call_retry_sleep_sec,
                api_call_delay_sec=api_call_delay_sec,
                concurrent_async_task=concurrent_async_task,
            )
        )
