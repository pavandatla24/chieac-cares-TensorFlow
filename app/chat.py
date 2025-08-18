from colorama import Fore, Style, init; init(autoreset=True)
from app.flow_engine import route_message, start_session
from app.safety import check_crisis
from app.i18n import detect_lang

def run():
    session = start_session()
    print(Fore.CYAN + "ChiEAC CARES (local). Type 'exit' to quit.")
    while True:
        user = input(Fore.GREEN + "you: ")
        if user.strip().lower() == "exit":
            print(Fore.CYAN + "bye. take care.")
            break
        lang = detect_lang(user) or session["lang"]
        crisis = check_crisis(user, lang)
        if crisis:
            print(Fore.RED + crisis)
            continue
        bot = route_message(user, session, lang)
        print(Fore.MAGENTA + f"bot: {bot}")

if __name__ == "__main__":
    run()
