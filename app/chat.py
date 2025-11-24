from colorama import Fore, Style, init; init(autoreset=True)
from app.flow_engine import route_message, start_session
from app.safety import check_crisis

def run():
    session = start_session()
    print(Fore.CYAN + "ChiEAC CARES (English). Type 'help' for tips, 'reset' to restart, 'exit' to quit.")
    # Show welcome message immediately
    print(Fore.MAGENTA + "bot: I'm here with you. Before we start: this is not medical care. If you're in danger or might harm yourself, reply 'HELP'. How can I help you today? You can try breathing exercises, grounding techniques, affirmations, or journal prompts. Or just tell me how you're feeling.")
    while True:
        user = input(Fore.GREEN + "you: ")
        if user.strip().lower() == "exit":
            print(Fore.CYAN + "bye. take care.")
            break
        if not user.strip():
            print(Fore.MAGENTA + "bot: I’m here. You can type breathing / grounding / affirmation / journal, or say how you feel.")
            continue
        if user.strip().lower() in ("help", "?", "tips"):
            print(Fore.MAGENTA + "bot: Try: 'breathing', 'grounding', 'affirmation', 'journal', or describe how you feel. Use 'reset' to restart a flow.")
            continue
        # English only - no language detection needed
        crisis = check_crisis(user, "en")
        if crisis:
            print(Fore.RED + crisis)
            continue
        bot = route_message(user, session, "en")
        print(Fore.MAGENTA + f"bot: {bot}")

if __name__ == "__main__":
    run()
