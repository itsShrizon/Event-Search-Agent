# event_search_agent.py
# A more dynamic event search assistant using LangChain's Tool Calling feature.

import os
import datetime
import logging
import difflib
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.render import render_text_description
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain.tools import tool
from pydantic import BaseModel, Field

# --- Basic Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
LLM_MODEL = "gpt-4o-mini"

# --- API Key Check ---
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# ==========================================================================
# 1. SIMULATED DATABASE MODELS & MOCK DATA (Unchanged)
# ==========================================================================

@dataclass
class Venue:
    id: int; name: str; city: str; country: str

@dataclass
class Event:
    id: int; event_name: str; venue_id: int

@dataclass
class Ticket:
    id: int; event_id: int; title: str; price: float; available_sales: int

mock_venues: List[Venue] = [
    # Thailand
    Venue(id=1, name="Upper House", city="Bangkok", country="Thailand"),
    Venue(id=2, name="Sky Lounge", city="Bangkok", country="Thailand"),
    Venue(id=3, name="Seaside Arena", city="Phuket", country="Thailand"),
    Venue(id=4, name="Chiang Mai Convention Center", city="Chiang Mai", country="Thailand"),

    # USA
    Venue(id=5, name="Madison Square Garden", city="New York", country="USA"),
    Venue(id=6, name="The Fillmore", city="San Francisco", country="USA"),
    Venue(id=7, name="Red Rocks Amphitheatre", city="Morrison", country="USA"),
    Venue(id=8, name="Austin City Limits Live", city="Austin", country="USA"),

    # UK
    Venue(id=9, name="The O2 Arena", city="London", country="UK"),
    Venue(id=10, name="Ministry of Sound", city="London", country="UK"),
    Venue(id=11, name="Albert Hall", city="Manchester", country="UK"),

    # Japan
    Venue(id=12, name="Tokyo Dome", city="Tokyo", country="Japan"),
    Venue(id=13, name="Shibuya Crossing Hall", city="Tokyo", country="Japan"),
    Venue(id=14, name="Osaka-jo Hall", city="Osaka", country="Japan"),
]

# More events, linked to the new venues
mock_events: List[Event] = [
    # Events in Thailand
    Event(id=101, event_name="Neon Night Party", venue_id=1),
    Event(id=102, event_name="Sunset Beats", venue_id=2),
    Event(id=103, event_name="Beach Fest 2025", venue_id=3),
    Event(id=104, event_name="Lanna Food Festival", venue_id=4),

    # Events in USA
    Event(id=105, event_name="Pop Superstars Live", venue_id=5),
    Event(id=106, event_name="Indie Rock Revival", venue_id=6),
    Event(id=107, event_name="Symphony Under the Stars", venue_id=7),
    Event(id=108, event_name="Country Music Showcase", venue_id=8),
    Event(id=109, event_name="Tech Conference 2025", venue_id=6), # Another event at The Fillmore

    # Events in UK
    Event(id=110, event_name="Global EDM Festival", venue_id=9),
    Event(id=111, event_name="House & Techno Night", venue_id=10),
    Event(id=112, event_name="The Smiths Tribute Concert", venue_id=11),
    Event(id=113, event_name="Comedy Gala Night", venue_id=9),

    # Events in Japan
    Event(id=114, event_name="J-Pop World Tour Final", venue_id=12),
    Event(id=115, event_name="Anime Convention Afterparty", venue_id=13),
    Event(id=116, event_name="International Jazz Summit", venue_id=14),
]

# More tickets with varied availability, prices, and types
mock_tickets: List[Ticket] = [
    # Tickets for Event 101 (Neon Night Party)
    Ticket(id=1001, event_id=101, title="General Admission", price=50.00, available_sales=150),
    Ticket(id=1002, event_id=101, title="VIP Access", price=120.00, available_sales=20),

    # Tickets for Event 102 (Sunset Beats)
    Ticket(id=1003, event_id=102, title="Rooftop Pass", price=75.00, available_sales=0), # Sold out

    # Tickets for Event 103 (Beach Fest 2025)
    Ticket(id=1004, event_id=103, title="Early Bird Beach Pass", price=55.00, available_sales=500),
    Ticket(id=1005, event_id=103, title="Standard Beach Pass", price=75.00, available_sales=1200),
    Ticket(id=1006, event_id=103, title="VIP Cabana", price=350.00, available_sales=15),

    # Tickets for Event 104 (Lanna Food Festival)
    Ticket(id=1007, event_id=104, title="Day Pass", price=25.00, available_sales=400),
    Ticket(id=1008, event_id=104, title="Tasting Coupon Bundle", price=40.00, available_sales=250),

    # Tickets for Event 105 (Pop Superstars Live)
    Ticket(id=1009, event_id=105, title="Upper Bowl Seating", price=89.99, available_sales=2500),
    Ticket(id=1010, event_id=105, title="Lower Bowl Seating", price=149.99, available_sales=1000),
    Ticket(id=1011, event_id=105, title="Floor Standing", price=199.99, available_sales=500),
    Ticket(id=1012, event_id=105, title="Meet & Greet Package", price=499.99, available_sales=10),

    # Tickets for Event 106 (Indie Rock Revival)
    Ticket(id=1013, event_id=106, title="General Admission", price=45.50, available_sales=300),

    # Tickets for Event 107 (Symphony Under the Stars)
    Ticket(id=1014, event_id=107, title="Lawn Ticket", price=35.00, available_sales=1000),
    Ticket(id=1015, event_id=107, title="Reserved Seating", price=95.00, available_sales=250),

    # Tickets for Event 110 (Global EDM Festival)
    Ticket(id=1016, event_id=110, title="Tier 1 - Early Bird", price=150.00, available_sales=0), # Sold out
    Ticket(id=1017, event_id=110, title="Tier 2 - General Sale", price=180.00, available_sales=450),
    Ticket(id=1018, event_id=110, title="VIP Pod", price=800.00, available_sales=30),

    # Tickets for Event 114 (J-Pop World Tour Final)
    Ticket(id=1019, event_id=114, title="S-Rank Seat (Lottery)", price=120.00, available_sales=500),
    Ticket(id=1020, event_id=114, title="A-Rank Seat", price=90.00, available_sales=3000),
    Ticket(id=1021, event_id=114, title="B-Rank Seat", price=70.00, available_sales=5000),

    # Note: Some events (108, 109, 111, 112, 113, 115, 116) have no tickets defined yet.
    # This can be useful for testing scenarios where events exist but tickets are not on sale.
]

# ==========================================================================
# 2. FUZZY MATCHING UTILITIES
# ==========================================================================

def fuzzy_match_builtin(query: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
    """
    Find the best fuzzy match using Python's built-in difflib.
    
    Args:
        query: The potentially misspelled string
        candidates: List of correct strings to match against
        threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        Best matching string or None if no match above threshold
    """
    if not candidates:
        return None
    
    # Get similarity ratios for all candidates
    matches = [(candidate, difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio()) 
              for candidate in candidates]
    
    # Find the best match
    best_match, best_score = max(matches, key=lambda x: x[1])
    
    # Return best match if it meets threshold
    return best_match if best_score >= threshold else None

def enhanced_fuzzy_search(
    query: str, 
    event_names: List[str], 
    venue_names: List[str], 
    ticket_types: List[str],
    threshold: float = 0.6
) -> Dict[str, Optional[str]]:
    """
    Perform fuzzy matching across different categories.
    
    Returns a dictionary with potential matches for each category.
    """
    results = {
        'event_match': fuzzy_match_builtin(query, event_names, threshold),
        'venue_match': fuzzy_match_builtin(query, venue_names, threshold),
        'ticket_match': fuzzy_match_builtin(query, ticket_types, threshold)
    }
    
    return results

# ==========================================================================
# 3. DEFINE THE ENHANCED DYNAMIC SEARCH TOOL
# ==========================================================================

class EventSearchInput(BaseModel):
    """Input model for the event search tool."""
    location_city: Optional[str] = Field(None, description="The city to search for events or venues in, e.g., 'Bangkok'.")
    location_country: Optional[str] = Field(None, description="The country to search for events or venues in, e.g., 'Thailand', 'UK', 'USA', 'Japan'.")
    venue_name: Optional[str] = Field(None, description="The specific name of the venue to search for, e.g., 'Upper House'.")
    event_name: Optional[str] = Field(None, description="The name of the event to search for, e.g., 'Pop Superstars Live'.")
    ticket_type: Optional[str] = Field(None, description="The specific ticket type to search for, e.g., 'VIP Access', 'General Admission', 'Upper Bowl Seating', 'Early Bird'.")
    aggregation: Optional[Literal['min', 'max', 'avg']] = Field(None, description="Perform a calculation on the ticket prices. Use 'min' for the cheapest price, 'max' for the most expensive.")
    include_sold_out: Optional[bool] = Field(True, description="Whether to include sold-out tickets in the results. Default is True.")
    fuzzy_threshold: Optional[float] = Field(0.6, description="Minimum similarity score for fuzzy matching (0.0 to 1.0). Default is 0.6.")

    
@tool(args_schema=EventSearchInput)
def search_events(
    location_city: Optional[str] = None, 
    location_country: Optional[str] = None, 
    venue_name: Optional[str] = None, 
    event_name: Optional[str] = None,
    ticket_type: Optional[str] = None,
    aggregation: Optional[str] = None,
    include_sold_out: Optional[bool] = True,
    fuzzy_threshold: Optional[float] = 0.6
) -> Dict[str, Any]:
    """
    Searches for venues, events, and tickets with fuzzy string matching for handling spelling mistakes.
    Can also perform calculations on prices.
    Use this tool to answer any questions about venues, events, tickets, and prices.
    
    To get ALL venues: Don't provide any location parameters.
    To search by city: Provide location_city (e.g., 'Bangkok', 'Phuket').
    To search by country: Provide location_country (e.g., 'Thailand', 'UK', 'USA', 'Japan').
    To search by venue: Provide venue_name (e.g., 'Upper House').
    To search by event: Provide event_name (e.g., 'Pop Superstars Live').
    To search for specific ticket types: Provide ticket_type (e.g., 'VIP Access', 'Upper Bowl Seating').
    For price calculations: Use aggregation ('min' for cheapest, 'max' for most expensive, 'avg' for average).
    Fuzzy matching helps with typos and approximate spellings.
    """
    logging.info(f"Tool 'search_events' called with: city='{location_city}', country='{location_country}', venue='{venue_name}', event='{event_name}', ticket='{ticket_type}', agg='{aggregation}', fuzzy_threshold={fuzzy_threshold}")
    
    # --- Fuzzy Matching Logic ---
    # Create lists for fuzzy matching
    all_event_names = [event.event_name for event in mock_events]
    all_venue_names = [venue.name for venue in mock_venues]
    all_ticket_types = list(set(ticket.title for ticket in mock_tickets))
    
    # Variables to track if fuzzy matching was used
    fuzzy_corrections = {}
    
    # Apply fuzzy matching for event_name
    if event_name:
        fuzzy_event = fuzzy_match_builtin(event_name, all_event_names, fuzzy_threshold)
        if fuzzy_event and fuzzy_event != event_name:
            fuzzy_corrections['event_name'] = {'original': event_name, 'corrected': fuzzy_event}
            event_name = fuzzy_event
        elif not fuzzy_event:
            # Try with lower threshold for suggestions
            suggestion = fuzzy_match_builtin(event_name, all_event_names, 0.3)
            if suggestion:
                return {
                    "status": "not_found_with_suggestion", 
                    "message": f"Event '{fuzzy_corrections.get('event_name', {}).get('original', event_name)}' not found. Did you mean '{suggestion}'?",
                    "suggestion": suggestion,
                    "original_query": fuzzy_corrections.get('event_name', {}).get('original', event_name)
                }
    
    # Apply fuzzy matching for venue_name
    if venue_name:
        fuzzy_venue = fuzzy_match_builtin(venue_name, all_venue_names, fuzzy_threshold)
        if fuzzy_venue and fuzzy_venue != venue_name:
            fuzzy_corrections['venue_name'] = {'original': venue_name, 'corrected': fuzzy_venue}
            venue_name = fuzzy_venue
        elif not fuzzy_venue:
            suggestion = fuzzy_match_builtin(venue_name, all_venue_names, 0.3)
            if suggestion:
                return {
                    "status": "not_found_with_suggestion", 
                    "message": f"Venue '{fuzzy_corrections.get('venue_name', {}).get('original', venue_name)}' not found. Did you mean '{suggestion}'?",
                    "suggestion": suggestion,
                    "original_query": fuzzy_corrections.get('venue_name', {}).get('original', venue_name)
                }
    
    # Apply fuzzy matching for ticket_type
    if ticket_type:
        fuzzy_ticket = fuzzy_match_builtin(ticket_type, all_ticket_types, fuzzy_threshold)
        if fuzzy_ticket and fuzzy_ticket != ticket_type:
            fuzzy_corrections['ticket_type'] = {'original': ticket_type, 'corrected': fuzzy_ticket}
            ticket_type = fuzzy_ticket
        elif not fuzzy_ticket:
            suggestion = fuzzy_match_builtin(ticket_type, all_ticket_types, 0.3)
            if suggestion:
                return {
                    "status": "not_found_with_suggestion", 
                    "message": f"Ticket type '{fuzzy_corrections.get('ticket_type', {}).get('original', ticket_type)}' not found. Did you mean '{suggestion}'?",
                    "suggestion": suggestion,
                    "original_query": fuzzy_corrections.get('ticket_type', {}).get('original', ticket_type)
                }

    # --- Database Filtering Logic ---
    candidate_venues = mock_venues
    if location_city:
        candidate_venues = [v for v in candidate_venues if location_city.lower() in v.city.lower()]
    if location_country:
        candidate_venues = [v for v in candidate_venues if location_country.lower() in v.country.lower()]
    if venue_name:
        candidate_venues = [v for v in candidate_venues if venue_name.lower() in v.name.lower()]

    # Filter events if event_name is provided
    candidate_events = mock_events
    if event_name:
        candidate_events = [e for e in candidate_events if event_name.lower() in e.event_name.lower()]

    # If we have specific events, only include venues that host these events
    if candidate_events != mock_events:
        venue_ids = set(e.venue_id for e in candidate_events)
        candidate_venues = [v for v in candidate_venues if v.id in venue_ids]

    if not candidate_venues:
        return {"status": "not_found", "message": "No venues found matching your criteria."}

    # --- IMPROVED Data Retrieval Logic ---
    results = []
    all_ticket_prices = []
    specific_ticket_matches = []
    sold_out_matches = []  # NEW: Track sold out tickets
    
    for venue in candidate_venues:
        events = [e for e in candidate_events if e.venue_id == venue.id]
        event_details = []
        
        for event in events:
            # CHANGED: Include all tickets, not just available ones
            all_tickets = [t for t in mock_tickets if t.event_id == event.id]
            available_tickets = [t for t in all_tickets if t.available_sales > 0]
            sold_out_tickets = [t for t in all_tickets if t.available_sales == 0]
            
            # Filter tickets by ticket_type if provided
            if ticket_type:
                matching_available = [t for t in available_tickets if ticket_type.lower() in t.title.lower()]
                matching_sold_out = [t for t in sold_out_tickets if ticket_type.lower() in t.title.lower()]
                
                # Store specific matches for both available and sold out
                for t in matching_available:
                    specific_ticket_matches.append({
                        "venue_name": venue.name,
                        "venue_location": f"{venue.city}, {venue.country}",
                        "event_name": event.event_name,
                        "ticket_title": t.title,
                        "price": t.price,
                        "available": t.available_sales,
                        "status": "available"
                    })
                
                for t in matching_sold_out:
                    sold_out_matches.append({
                        "venue_name": venue.name,
                        "venue_location": f"{venue.city}, {venue.country}",
                        "event_name": event.event_name,
                        "ticket_title": t.title,
                        "price": t.price,
                        "available": t.available_sales,
                        "status": "sold_out"
                    })
                
                # Use matching tickets for display
                if include_sold_out:
                    display_tickets = matching_available + matching_sold_out
                else:
                    display_tickets = matching_available
            else:
                # No specific ticket type filter
                if include_sold_out:
                    display_tickets = available_tickets + sold_out_tickets
                else:
                    display_tickets = available_tickets
            
            # Create ticket info with status
            if display_tickets:
                ticket_info = []
                for t in display_tickets:
                    status = "available" if t.available_sales > 0 else "sold_out"
                    ticket_info.append({
                        "title": t.title, 
                        "price": t.price, 
                        "available": t.available_sales,
                        "status": status
                    })
                
                # Only add available ticket prices to aggregation calculations
                available_prices = [t.price for t in display_tickets if t.available_sales > 0]
                all_ticket_prices.extend(available_prices)
                
                event_details.append({"event_name": event.event_name, "tickets": ticket_info})
            else:
                # Include events even if no tickets match
                event_details.append({"event_name": event.event_name, "tickets": []})
        
        if event_details:
            results.append({
                "venue_name": venue.name, 
                "location": f"{venue.city}, {venue.country}", 
                "country": venue.country, 
                "events": event_details
            })

    # IMPROVED: Handle specific ticket type queries with better messaging
    if ticket_type and event_name:
        if specific_ticket_matches and sold_out_matches:
            result = {
                "status": "success", 
                "message": f"Found {ticket_type} tickets for {event_name}. Some tickets are available, others are sold out.",
                "available_matches": specific_ticket_matches,
                "sold_out_matches": sold_out_matches,
                "data": results
            }
        elif specific_ticket_matches:
            result = {
                "status": "success", 
                "message": f"Found available {ticket_type} tickets for {event_name}.",
                "available_matches": specific_ticket_matches,
                "data": results
            }
        elif sold_out_matches:
            result = {
                "status": "sold_out", 
                "message": f"{ticket_type} tickets for {event_name} are currently sold out, but were priced at ${sold_out_matches[0]['price']:.2f}.",
                "sold_out_matches": sold_out_matches,
                "data": results
            }
        else:
            result = {
                "status": "not_found", 
                "message": f"No {ticket_type} tickets found for {event_name}.",
                "data": results
            }
        
        # Add fuzzy corrections to the response if any were made
        if fuzzy_corrections:
            result["corrections"] = fuzzy_corrections
            result["status"] = "success_with_corrections" if result["status"] == "success" else result["status"]
            
        return result
            
    if not results:
        # Check if we have corrections to mention
        if fuzzy_corrections:
            return {
                "status": "not_found_with_corrections", 
                "message": "No venues or events found matching your criteria, even with spelling corrections.",
                "corrections": fuzzy_corrections
            }
        else:
            return {"status": "not_found", "message": "No venues or events found matching your criteria."}
    

    # --- Aggregation Logic ---
    if aggregation and all_ticket_prices:
        if aggregation == 'min':
            value = min(all_ticket_prices)
            return {"status": "success", "message": f"The minimum ticket price is ${value:.2f}.", "data": results}
        if aggregation == 'max':
            value = max(all_ticket_prices)
            return {"status": "success", "message": f"The maximum ticket price is ${value:.2f}.", "data": results}
        if aggregation == 'avg':
            value = sum(all_ticket_prices) / len(all_ticket_prices)
            return {"status": "success", "message": f"The average ticket price is ${value:.2f}.", "data": results}

    # Return final results with corrections if any were made
    final_result = {"status": "success", "data": results}
    if fuzzy_corrections:
        final_result["corrections"] = fuzzy_corrections
        final_result["status"] = "success_with_corrections"
        final_result["message"] = "Results found using spelling corrections."
    
    return final_result

# ==========================================================================
# 4. MAIN ORCHESTRATOR (AGENTIC WORKFLOW)
# ==========================================================================

def process_user_query(query: str, chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
    """
    Orchestrates the AI agent process.
    The AI decides whether to call the search_events tool or respond directly.
    """
    chat_history = chat_history or []
    
    # 1. Initialize the LLM and bind our tool to it
    tools = [search_events]
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.5)  # Lower temperature for more consistent responses
    
    # Generate tool descriptions using render_text_description
    tool_descriptions = render_text_description(tools)
    logging.info(f"Tool descriptions: {tool_descriptions}")
    
    # Create a parser for structured tool outputs
    parser = PydanticToolsParser(tools=tools)
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Enhanced system prompt using the SystemMessage class directly
    system_content = f"""You are an intelligent event ticketing assistant with access to a comprehensive event database. Your goal is to provide helpful, detailed, and conversational responses about events, venues, and tickets.

CAPABILITIES:
- Search for venues by city, country, or name
- Find events at specific venues
- Get ticket information and pricing (including sold-out tickets)
- Search for specific ticket types by name
- Calculate min/max/average prices for available tickets
- Provide recommendations and insights
- Handle spelling mistakes and typos with fuzzy matching

IMPORTANT GUIDELINES:
- NEVER invent or make up any information that isn't in the database
- ALWAYS extract exact ticket prices directly from the data provided
- DO NOT use aggregation functions (min/max/avg) unless specifically asked about price ranges
- When asked about a specific ticket type, use event_name and ticket_type parameters to find the exact price
- Provide accurate information - avoid guessing or generalizing
- If you don't know something, say so rather than making it up

HANDLING FUZZY MATCHING & CORRECTIONS:
- When tool returns "corrections" field, acknowledge the spelling correction helpfully
- For "not_found_with_suggestion" status, ask if they meant the suggested spelling
- For "success_with_corrections" status, mention that you found results despite the typo
- Be encouraging about typos - everyone makes them!

HANDLING SOLD-OUT TICKETS:
- When tool returns "sold_out_matches", inform users that tickets exist but are sold out
- Always mention the original price of sold-out tickets
- Suggest alternatives or ask if they want to see other available tickets
- Differentiate clearly between "not available" (sold out) vs "doesn't exist"

QUERY UNDERSTANDING:
- Questions like "What is the price of Upper Bowl Seating for Pop Superstars Live?" should use:
  * event_name="Pop Superstars Live"
  * ticket_type="Upper Bowl Seating"
- For "What events are in UK?" use location_country="UK"
- For "Show me all venues in Bangkok" use location_city="Bangkok"
- For early bird or specific ticket inquiries, always use ticket_type parameter

RESPONSE HANDLING BASED ON TOOL STATUS:
- status="success": Normal response with available data
- status="success_with_corrections": Acknowledge spelling corrections positively
- status="not_found_with_suggestion": Ask if they meant the suggested spelling
- status="sold_out": Emphasize that tickets exist but are sold out, mention original price
- status="not_found": Clearly state what wasn't found and suggest alternatives
- Handle both "available_matches" and "sold_out_matches" in responses

LOCATION HANDLING:
- When users mention countries (e.g., "UK", "USA", "Thailand", "Japan") - use location_country parameter
- When users mention specific cities (e.g., "Bangkok", "London", "Tokyo") - use location_city parameter
- For budget questions, search all events and filter the results to show options within their price range

INSTRUCTIONS:
- Always be proactive and helpful
- Use the search_events tool to get accurate data about venues, events, and tickets
- Respond based ONLY on the actual tool results - never fabricate information
- If asking for "all venues" or general information, call search_events with no parameters
- For price queries about specific tickets, use event_name and ticket_type parameters
- If tool returns no results, suggest alternatives or ask for clarification
- Be specific and detailed in your responses
- When tickets are sold out, be empathetic but also suggest alternatives

EXAMPLES:
- "What is the price of Upper Bowl Seating for Pop Superstars Live?" → Use search_events with event_name="Pop Superstars Live", ticket_type="Upper Bowl Seating"
- "What events are in UK?" → Use search_events with location_country="UK"
- "Events in Bangkok?" → Use search_events with location_city="Bangkok"  
- "What's the cheapest ticket?" → Use search_events with aggregation="min"
- "Show me all venues" → Use search_events with no parameters
- "Early bird tickets for Global EDM Festival?" → Use search_events with event_name="Global EDM Festival", ticket_type="Early Bird"

AVAILABLE TOOLS:
{tool_descriptions}
"""
    system_message = SystemMessage(content=system_content)
    
    # 3. Create messages with properly constructed message objects
    messages = [system_message]
    
    # Add historical messages
    for msg_type, msg_content in chat_history:
        if msg_type == 'human':
            messages.append(HumanMessage(content=msg_content))
        else:
            messages.append(AIMessage(content=str(msg_content)))
    
    # Add the current query
    messages.append(HumanMessage(content=query))
    
    # 4. Invoke the LLM with the message objects directly
    logging.info("Invoking AI with enhanced tool-calling capabilities...")
    ai_response = llm_with_tools.invoke(messages)
    
    # 5. Check if the AI wants to call a tool
    if not ai_response.tool_calls:
        logging.info("AI responded directly without using a tool.")
        return {'status': 'success', 'message': ai_response.content, 'type': 'direct_response'}
    
    # 6. Execute the tool call and generate a comprehensive response
    logging.info(f"AI wants to call a tool: {ai_response.tool_calls}")
    
    # Use PydanticToolsParser to parse the structured tool calls if needed
    try:
        tool_map = {tool.name: tool for tool in tools}
        
        # Execute the first tool call
        call = ai_response.tool_calls[0]
        tool_to_call = tool_map.get(call['name'])
        
        if not tool_to_call:
            return {'status': 'error', 'message': f"AI tried to call a non-existent tool: {call['name']}"}
        
        tool_result = tool_to_call.invoke(call['args'])
        
        # 7. Generate a natural language response based on tool results using SystemMessage directly
        response_system_content = """You are an event assistant. Based on the tool results provided, give a helpful, detailed, and conversational response to the user's question. 

GUIDELINES:
- Be specific and informative
- Format information clearly (use bullet points, numbers, etc. when helpful)
- Include relevant details like prices, availability, locations
- NEVER invent information that isn't in the tool results
- ALWAYS use the exact price values from the tool results - do not round or average
- For specific price queries, quote the EXACT price found in the data
- If specific ticket matches are found in the results, prioritize showing those exact matches
- Check for "specific_matches" in the tool results for exact ticket information
- If no results found, suggest alternatives
- Always end with a helpful question or suggestion
- Be enthusiastic and engaging
- Don't mention "tool results" - speak naturally as if you know this information"""

        response_messages = [
            SystemMessage(content=response_system_content),
            HumanMessage(content=f"User asked: {query}"),
            HumanMessage(content=f"Tool results: {str(tool_result)}"),
            HumanMessage(content="Please provide a comprehensive and helpful response:")
        ]
        
        response_chain = llm
        final_response = response_chain.invoke(response_messages)
        
        # 8. Return the enhanced result
        return {
            'status': 'success', 
            'message': final_response.content,
            'type': 'tool_assisted_response',
            'tool_data': tool_result  # Include raw data for potential use
        }
        
    except Exception as e:
        logging.error(f"Error processing tool call: {e}")
        return {
            'status': 'error',
            'message': f"An error occurred while processing your request: {str(e)}",
            'type': 'error'
        }