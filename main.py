from register import register_person
from recognize import recognize_person


def show_menu():
    print("\n=== FacialBiometry ===")
    print("1 - Register person")
    print("2 - Recognize person")
    print("3 - Exit")


while True:

    show_menu()

    option = input("Choose an option: ")

    if option == "1":
        register_person()

    elif option == "2":
        recognize_person()

    elif option == "3":
        print("Closing system...")
        break

    else:
        print("Invalid option.")