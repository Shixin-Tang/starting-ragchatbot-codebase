from openai import OpenAI
from typing import List, Optional, Dict, Any, Union, Callable
import json
from enum import Enum
from dataclasses import dataclass, field
import time
import logging
from abc import ABC, abstractmethod

# Core classes for state-machine based sequential processing

class ConversationState(Enum):
    """States in the conversation processing state machine"""
    INITIAL = "initial"
    TOOL_EXECUTION = "tool_execution"
    REASONING = "reasoning"
    FINAL_RESPONSE = "final_response"
    ERROR_RECOVERY = "error_recovery"

class QueryIntent(Enum):
    """Types of user query intents"""
    CONTENT_SEARCH = "content_search"
    OUTLINE_REQUEST = "outline_request"
    COMPARISON = "comparison"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MULTI_STEP = "multi_step"
    UNKNOWN = "unknown"

@dataclass
class ProcessingResult:
    """Result of processing a query"""
    content: str
    rounds_used: int
    tools_executed: List[str]
    termination_reason: str
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    error_recovery_attempts: int = 0

@dataclass
class ConversationContext:
    """Rich conversation state across rounds"""
    query: str
    round: int = 0
    max_rounds: int = 2
    state: ConversationState = ConversationState.INITIAL
    intent: QueryIntent = QueryIntent.UNKNOWN
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_execution_history: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    error_recovery_attempts: int = 0
    start_time: float = field(default_factory=time.time)
    all_tools: List[Dict[str, Any]] = field(default_factory=list)
    
    def is_complete(self) -> bool:
        """Check if conversation should be considered complete"""
        return (
            self.state == ConversationState.FINAL_RESPONSE or
            self.round >= self.max_rounds or
            self.state == ConversationState.ERROR_RECOVERY and self.error_recovery_attempts > 2
        )
    
    def get_previous_tool_results(self) -> List[Dict[str, Any]]:
        """Get all tool results from previous rounds"""
        return [entry for entry in self.tool_execution_history if entry.get('result')]
    
    def has_relevant_content(self) -> bool:
        """Check if we have found relevant content for the query"""
        for result in self.get_previous_tool_results():
            if result.get('result') and 'no results' not in result['result'].lower():
                return True
        return False
    
    def has_complete_outline(self) -> bool:
        """Check if we have a complete course outline"""
        for result in self.get_previous_tool_results():
            if result.get('tool_name') == 'get_course_outline' and result.get('result'):
                return 'Course Title:' in result['result'] or 'Lesson' in result['result']
        return False
    
    def comparison_completeness_score(self) -> float:
        """Calculate completeness score for comparison queries"""
        results = self.get_previous_tool_results()
        if len(results) >= 2:
            return 1.0
        elif len(results) == 1:
            return 0.5
        return 0.0

class AdaptiveToolPolicy:
    """Manages tool availability based on conversation context"""
    
    def determine_available_tools(self, context: ConversationContext) -> List[Dict]:
        """Dynamically filter tools based on context and previous results"""
        if not context.all_tools:
            return []
            
        available_tools = []
        previous_results = context.get_previous_tool_results()
        
        # Round 1: All tools available
        if context.round == 1:
            available_tools = context.all_tools.copy()
            
        # Round 2: Filter based on previous results and intent
        elif context.round == 2:
            if context.intent == QueryIntent.COMPARISON:
                # For comparisons, allow search tools for additional content
                available_tools = [t for t in context.all_tools 
                                 if t['name'] == 'search_course_content']
            
            elif context.intent == QueryIntent.MULTI_STEP:
                # For multi-step queries, enable complementary tools
                search_used = any(r.get('tool_name') == 'search_course_content' for r in previous_results)
                outline_used = any(r.get('tool_name') == 'get_course_outline' for r in previous_results)
                
                if search_used and not outline_used:
                    available_tools = [t for t in context.all_tools 
                                     if t['name'] == 'get_course_outline']
                elif outline_used and not search_used:
                    available_tools = [t for t in context.all_tools 
                                     if t['name'] == 'search_course_content']
            
            elif self._previous_search_empty(previous_results):
                # If search returned no results, try outline tool
                available_tools = [t for t in context.all_tools 
                                 if t['name'] == 'get_course_outline']
            
            elif self._previous_search_successful(previous_results):
                # If search was successful, enable synthesis mode (no tools)
                available_tools = []
        
        return available_tools
    
    def _previous_search_empty(self, results: List[Dict[str, Any]]) -> bool:
        """Check if previous search returned empty results"""
        for result in results:
            if (result.get('tool_name') == 'search_course_content' and 
                result.get('result') and 
                ('no results' in result['result'].lower() or len(result['result'].strip()) < 50)):
                return True
        return False
    
    def _previous_search_successful(self, results: List[Dict[str, Any]]) -> bool:
        """Check if previous search was successful"""
        for result in results:
            if (result.get('tool_name') == 'search_course_content' and 
                result.get('result') and 
                'no results' not in result['result'].lower() and 
                len(result['result'].strip()) > 100):
                return True
        return False
    
    def classify_intent(self, query: str) -> QueryIntent:
        """Analyze query to understand user intent"""
        query_lower = query.lower()
        
        # Look for comparison keywords
        comparison_keywords = ['compare', 'difference', 'vs', 'versus', 'between']
        if any(kw in query_lower for kw in comparison_keywords):
            return QueryIntent.COMPARISON
        
        # Look for outline requests
        outline_keywords = ['outline', 'structure', 'overview', 'table of contents', 'lessons']
        if any(kw in query_lower for kw in outline_keywords):
            return QueryIntent.OUTLINE_REQUEST
        
        # Look for specific content searches
        content_keywords = ['lesson', 'chapter', 'section', 'how to', 'what is', 'explain']
        course_indicators = ['course', 'mcp', 'tutorial']
        if (any(kw in query_lower for kw in content_keywords) and 
            any(ci in query_lower for ci in course_indicators)):
            return QueryIntent.CONTENT_SEARCH
        
        # Look for multi-step indicators
        multi_step_keywords = ['then', 'after that', 'next', 'also show', 'and then']
        if any(kw in query_lower for kw in multi_step_keywords):
            return QueryIntent.MULTI_STEP
        
        # Check if query seems course-related at all
        if any(ci in query_lower for ci in course_indicators):
            return QueryIntent.CONTENT_SEARCH
        
        # Default to general knowledge
        return QueryIntent.GENERAL_KNOWLEDGE

class TerminationManager:
    """Intelligent termination based on multiple criteria"""
    
    def should_terminate(self, context: ConversationContext, last_action_result: Dict[str, Any]) -> tuple[bool, str]:
        """Multi-factor termination decision"""
        
        # Hard limits
        if context.round >= context.max_rounds:
            return True, "max_rounds_reached"
        
        if context.error_recovery_attempts > 2:
            return True, "error_threshold_exceeded"
        
        # Check if no tools were used in last round (natural completion)
        if (last_action_result.get('action_type') == 'reasoning' and 
            not last_action_result.get('tool_calls')):
            return True, "natural_completion"
        
        # Multi-criteria scoring
        criteria_scores = [
            self._information_completeness(context),
            self._user_satisfaction_threshold(context),
            self._cost_benefit_analysis(context),
            self._diminishing_returns_detected(context)
        ]
        
        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_score = sum(w * score for w, score in zip(weights, criteria_scores))
        
        if weighted_score > 0.75:
            return True, "information_complete"
        
        return False, "continue"
    
    def _information_completeness(self, context: ConversationContext) -> float:
        """Assess if we have sufficient information"""
        if context.intent == QueryIntent.CONTENT_SEARCH:
            return 1.0 if context.has_relevant_content() else 0.0
        elif context.intent == QueryIntent.OUTLINE_REQUEST:
            return 1.0 if context.has_complete_outline() else 0.0
        elif context.intent == QueryIntent.COMPARISON:
            return context.comparison_completeness_score()
        elif context.intent == QueryIntent.GENERAL_KNOWLEDGE:
            return 1.0  # General knowledge doesn't need tools
        return 0.5  # Default for unclear intent
    
    def _user_satisfaction_threshold(self, context: ConversationContext) -> float:
        """Estimate user satisfaction based on content quality"""
        results = context.get_previous_tool_results()
        if not results:
            return 0.0
        
        # Check quality of results
        quality_score = 0.0
        for result in results:
            result_text = result.get('result', '')
            if len(result_text) > 200:  # Substantial content
                quality_score += 0.5
            if 'no results' not in result_text.lower():  # Not empty
                quality_score += 0.3
        
        return min(1.0, quality_score)
    
    def _cost_benefit_analysis(self, context: ConversationContext) -> float:
        """Balance information gain vs resource usage"""
        if context.round == 1:
            return 0.0  # First round is always worth it
        
        # Check if second round added significant value
        first_round_results = [r for r in context.tool_execution_history if r.get('round') == 1]
        second_round_results = [r for r in context.tool_execution_history if r.get('round') == 2]
        
        if not second_round_results:
            return 0.0
        
        # If second round found new information, it was worth it
        for result in second_round_results:
            if result.get('result') and len(result['result']) > 100:
                return 0.8
        
        return 0.3  # Some value but not significant
    
    def _diminishing_returns_detected(self, context: ConversationContext) -> float:
        """Detect if additional rounds are providing diminishing returns"""
        if context.round < 2:
            return 0.0
        
        results = context.get_previous_tool_results()
        if len(results) < 2:
            return 0.0
        
        # Check if recent results are similar to previous ones
        recent_result = results[-1].get('result', '')
        previous_results = [r.get('result', '') for r in results[:-1]]
        
        # Simple similarity check (could be enhanced with embeddings)
        for prev_result in previous_results:
            if len(recent_result) > 0 and len(prev_result) > 0:
                common_words = set(recent_result.lower().split()) & set(prev_result.lower().split())
                similarity = len(common_words) / max(len(recent_result.split()), len(prev_result.split()))
                if similarity > 0.7:  # High similarity suggests diminishing returns
                    return 1.0
        
        return 0.0

class ErrorRecoveryManager:
    """Handles errors with intelligent recovery strategies"""
    
    def handle_error(self, error: Exception, context: ConversationContext) -> ConversationContext:
        """Context-aware error recovery"""
        context.error_recovery_attempts += 1
        context.state = ConversationState.ERROR_RECOVERY
        
        # Log the error
        logging.warning(f"Error in conversation round {context.round}: {error}")
        
        # Add error information to semantic context
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'round': context.round,
            'recovery_attempt': context.error_recovery_attempts
        }
        
        if 'errors' not in context.semantic_context:
            context.semantic_context['errors'] = []
        context.semantic_context['errors'].append(error_info)
        
        # Continue processing but mark as degraded
        context.state = ConversationState.REASONING
        return context

class SequentialAIProcessor:
    """State-machine based processor for multi-round tool calling"""
    
    def __init__(self, api_key: str, model: str, max_rounds: int = 2):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_rounds = max_rounds
        self.tool_policy = AdaptiveToolPolicy()
        self.termination_manager = TerminationManager()
        self.recovery_manager = ErrorRecoveryManager()
        
        # Base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def process_query(self, query: str, tools: List[Dict], tool_manager, 
                     conversation_history: Optional[str] = None) -> ProcessingResult:
        """Event-driven multi-round processing"""
        
        # Initialize conversation context
        context = ConversationContext(
            query=query,
            max_rounds=self.max_rounds,
            all_tools=tools.copy() if tools else []
        )
        
        # Classify intent
        context.intent = self.tool_policy.classify_intent(query)
        
        # Build initial messages
        context.messages = self._build_initial_messages(query, conversation_history)
        
        action_result = {}
        
        # Main processing loop
        while not context.is_complete():
            try:
                context.round += 1
                context.state = ConversationState.TOOL_EXECUTION
                
                # Determine available tools for this round
                available_tools = self.tool_policy.determine_available_tools(context)
                
                # Execute round
                action_result = self._execute_round(context, available_tools, tool_manager)
                
                # Check termination conditions
                should_terminate, reason = self.termination_manager.should_terminate(context, action_result)
                if should_terminate:
                    break
                    
            except Exception as e:
                context = self.recovery_manager.handle_error(e, context)
                action_result = {'action_type': 'error_recovery', 'error': str(e)}
        
        # Finalize response
        return self._finalize_response(context, action_result)
    
    def _build_initial_messages(self, query: str, conversation_history: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build initial message array for the conversation"""
        messages = []
        
        # Enhanced system prompt for sequential processing
        system_content = self._build_system_prompt(conversation_history)
        messages.append({"role": "system", "content": system_content})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _build_system_prompt(self, conversation_history: Optional[str] = None) -> str:
        """Build enhanced system prompt for sequential processing"""
        base_prompt = """You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: For searching specific course content and detailed educational materials
2. **get_course_outline**: For retrieving course outlines with title, course link, and complete lesson lists

Sequential Tool Usage:
- You can make tool calls across multiple rounds of reasoning (max 2 rounds)
- Round 1: Make initial tool calls to gather information
- Round 2: If needed, make additional tool calls based on Round 1 results to complete your understanding
- Build upon previous tool results to provide comprehensive answers
- Examples requiring multiple rounds:
  * Comparing content from different lessons or courses
  * Getting course outline then searching for specific lesson content
  * Searching for related topics across multiple courses

Tool Usage Guidelines:
- Use **search_course_content** for questions about specific course content or detailed educational materials
- Use **get_course_outline** for questions about course structure, lesson lists, or when users want to see what's covered in a course
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Use search_course_content tool first, then answer
- **Course outline/structure questions**: Use get_course_outline tool first, then answer
- **Complex queries**: Use multiple rounds as needed to gather complete information
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
  - Do not mention "based on the search results" or "based on the outline"

For outline queries, always include:
- Course title
- Course link
- Complete lesson list with lesson numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked."""
        
        if conversation_history:
            return f"{base_prompt}\n\nPrevious conversation:\n{conversation_history}"
        return base_prompt
    
    def _execute_round(self, context: ConversationContext, available_tools: List[Dict], 
                      tool_manager) -> Dict[str, Any]:
        """Execute a single round of conversation"""
        
        # Prepare API parameters
        api_params = {
            **self.base_params,
            "messages": context.messages.copy()
        }
        
        # Add tools if available
        if available_tools:
            openai_tools = self._convert_tools_to_openai_format(available_tools)
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"
        
        # Get response from OpenAI
        response = self.client.chat.completions.create(**api_params)
        
        # Process response
        if response.choices[0].message.tool_calls and tool_manager:
            return self._handle_tool_execution(response, context, tool_manager)
        else:
            # Direct response without tools
            context.state = ConversationState.REASONING
            context.messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
            return {
                "action_type": "reasoning",
                "content": response.choices[0].message.content,
                "tool_calls": False
            }
    
    def _handle_tool_execution(self, response, context: ConversationContext, tool_manager) -> Dict[str, Any]:
        """Handle tool execution and update context"""
        
        # Add assistant message with tool calls
        assistant_message = {
            "role": "assistant",
            "content": response.choices[0].message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in response.choices[0].message.tool_calls
            ]
        }
        context.messages.append(assistant_message)
        
        tools_executed = []
        
        # Execute all tool calls
        for tool_call in response.choices[0].message.tool_calls:
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            
            # Execute the tool
            try:
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name, 
                    **tool_args
                )
                tools_executed.append(tool_call.function.name)
            except Exception as e:
                tool_result = f"Error executing tool: {str(e)}"
            
            # Truncate if too long
            if len(tool_result) > 2000:
                tool_result = tool_result[:2000] + "... [truncated]"
            
            # Add tool result to messages
            context.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
            
            # Track in execution history
            context.tool_execution_history.append({
                "round": context.round,
                "tool_name": tool_call.function.name,
                "arguments": tool_args,
                "result": tool_result,
                "timestamp": time.time()
            })
        
        # Get follow-up response if this is not the final round
        if context.round < context.max_rounds:
            # Continue conversation in next round
            return {
                "action_type": "tool_execution",
                "tools_executed": tools_executed,
                "tool_calls": True
            }
        else:
            # Final round - get synthesis response
            final_params = {
                **self.base_params,
                "messages": context.messages
            }
            final_response = self.client.chat.completions.create(**final_params)
            
            context.messages.append({
                "role": "assistant",
                "content": final_response.choices[0].message.content
            })
            
            return {
                "action_type": "final_synthesis",
                "content": final_response.choices[0].message.content,
                "tools_executed": tools_executed,
                "tool_calls": True
            }
    
    def _convert_tools_to_openai_format(self, anthropic_tools: List[Dict]) -> List[Dict]:
        """Convert Anthropic tool format to OpenAI function calling format"""
        openai_tools = []
        for tool in anthropic_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools
    
    def _finalize_response(self, context: ConversationContext, last_action: Dict[str, Any]) -> ProcessingResult:
        """Create final processing result"""
        
        # Get final content
        final_content = ""
        if context.messages and context.messages[-1]["role"] == "assistant":
            final_content = context.messages[-1]["content"]
        
        # Determine termination reason
        if context.error_recovery_attempts > 0:
            termination_reason = "error_recovery"
        elif context.round >= context.max_rounds:
            termination_reason = "max_rounds"
        elif last_action.get('action_type') == 'reasoning':
            termination_reason = "natural_completion"
        else:
            termination_reason = "information_complete"
        
        # Collect all executed tools
        tools_executed = list(set([
            entry['tool_name'] for entry in context.tool_execution_history
        ]))
        
        return ProcessingResult(
            content=final_content or "Unable to generate response",
            rounds_used=context.round,
            tools_executed=tools_executed,
            termination_reason=termination_reason,
            semantic_context=context.semantic_context,
            error_recovery_attempts=context.error_recovery_attempts
        )

class AIGenerator:
    """Handles interactions with OpenAI's GPT API for generating responses with sequential tool calling"""
    
    # Legacy system prompt for backward compatibility
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: For searching specific course content and detailed educational materials
2. **get_course_outline**: For retrieving course outlines with title, course link, and complete lesson lists

Tool Usage Guidelines:
- Use **search_course_content** for questions about specific course content or detailed educational materials
- Use **get_course_outline** for questions about course structure, lesson lists, or when users want to see what's covered in a course
- **One tool call per query maximum**
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Use search_course_content tool first, then answer
- **Course outline/structure questions**: Use get_course_outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"

For outline queries, always include:
- Course title
- Course link
- Complete lesson list with lesson numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, enable_sequential: bool = True):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.enable_sequential = enable_sequential
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        
        # Initialize sequential processor
        if self.enable_sequential:
            self.sequential_processor = SequentialAIProcessor(api_key, model, max_rounds=2)
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling for complex queries.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Use sequential processor if enabled and tools are available
        if self.enable_sequential and tools and tool_manager:
            try:
                result = self.sequential_processor.process_query(
                    query=query,
                    tools=tools,
                    tool_manager=tool_manager,
                    conversation_history=conversation_history
                )
                return result.content
            except Exception as e:
                # Fallback to legacy mode on error
                logging.warning(f"Sequential processing failed, falling back to legacy mode: {e}")
                return self._legacy_generate_response(query, conversation_history, tools, tool_manager)
        else:
            # Use legacy single-round processing
            return self._legacy_generate_response(query, conversation_history, tools, tool_manager)
    
    def _legacy_generate_response(self, query: str,
                                 conversation_history: Optional[str] = None,
                                 tools: Optional[List] = None,
                                 tool_manager=None) -> str:
        """
        Legacy single-round response generation for backward compatibility.
        """
        
        # Build messages array for OpenAI format
        messages = []
        
        # Add system message
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        messages.append({"role": "system", "content": system_content})
        
        # Add user message
        messages.append({"role": "user", "content": query})
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Add tools if available (convert to OpenAI format)
        if tools:
            openai_tools = self._convert_tools_to_openai_format(tools)
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"
        
        # Get response from OpenAI
        response = self.client.chat.completions.create(**api_params)
        
        # Handle tool execution if needed
        if response.choices[0].message.tool_calls and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.choices[0].message.content
    
    def _convert_tools_to_openai_format(self, anthropic_tools: List[Dict]) -> List[Dict]:
        """Convert Anthropic tool format to OpenAI function calling format"""
        openai_tools = []
        for tool in anthropic_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's response with tool calls
        assistant_message = {
            "role": "assistant",
            "content": initial_response.choices[0].message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in initial_response.choices[0].message.tool_calls
            ]
        }
        messages.append(assistant_message)
        
        # Execute all tool calls and add results
        for tool_call in initial_response.choices[0].message.tool_calls:
            # Parse tool arguments
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            
            # Execute the tool
            try:
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name, 
                    **tool_args
                )
            except Exception as e:
                tool_result = f"Error executing tool: {str(e)}"
            
            # Truncate tool result if too long to avoid token limits
            if len(tool_result) > 2000:
                tool_result = tool_result[:2000] + "... [truncated]"
            
            # Add tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content