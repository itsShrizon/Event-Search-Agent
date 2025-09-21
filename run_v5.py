# run.py
# An interactive command-line script to run the AI Event Assistant.

import pprint
from typing import List, Tuple

# MODIFIED IMPORT: Import from the new agent file
from ai_logic_v5 import process_user_query

def main():
    """
    Main function to run the interactive chat loop.
    """
    print("=" * 60)
    print("🤖 AI Event Ticketing Assistant (v2 - Dynamic)")
    print("=" * 60)
    print("Try asking me complex questions like:")
    print("- 'What is the cheapest ticket at Upper House in Bangkok?'")
    print("- 'Show me the prices for venues in Phuket.'")
    print("- 'Are there any venues in Thailand?'")
    print("=" * 60)
    
    chat_history: List[Tuple[str, str]] = []

    while True:
        try:
            user_query = input("\n👤 You: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for using the AI Event Assistant!")
                break
                
            if not user_query:
                continue

            print("-" * 60)
            print(f"🔍 Processing query: '{user_query}'")
            
            final_results = process_user_query(user_query, chat_history)
            
            message_for_display = ""
            message_for_history = ""

            if final_results.get('type') == 'casual_chat':
                message_for_display = f"💭 {final_results.get('message', '...')}"
                message_for_history = final_results.get('message', '')

            elif final_results.get('status') == 'success':
                # Success can now be a direct message (from aggregation) or data
                message_for_display = f"✅ {final_results.get('message', 'Results found!')}"
                if 'data' in final_results:
                    pretty_data = pprint.pformat(final_results['data'])
                    message_for_display += f"\n{pretty_data}"
                    message_for_history = pretty_data
                else:
                    message_for_history = final_results.get('message', '') # For min/max results
            else:
                message_for_display = f"⚠️ {final_results.get('message', 'Something went wrong.')}"
                message_for_history = final_results.get('message', '')

            print(f"\n🤖 AI Assistant:\n{message_for_display}")
            print("-" * 60)
            
            chat_history.append(("human", user_query))
            chat_history.append(("ai", message_for_history))
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the AI Event Assistant!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue

if __name__ == "__main__":
    main()