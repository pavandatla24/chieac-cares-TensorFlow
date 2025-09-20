from colorama import Fore, Style, init; init(autoreset=True)
from app.flow_engine import route_message, start_session
from app.safety import check_crisis

def run():
    session = start_session()
    print(Fore.CYAN + "ChiEAC CARES (English). Type 'exit' to quit.")
    while True:
        user = input(Fore.GREEN + "you: ")
        if user.strip().lower() == "exit":
            print(Fore.CYAN + "bye. take care.")
            break
        # English only - no language detection needed
        crisis = check_crisis(user, "en")
        if crisis:
            print(Fore.RED + crisis)
            continue
        bot = route_message(user, session, "en")
        print(Fore.MAGENTA + f"bot: {bot}")

if __name__ == "__main__":
    run()
