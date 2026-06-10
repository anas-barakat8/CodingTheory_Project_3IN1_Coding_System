# This file will start the experiments


from experiments import *


def main():
    while True:
        print("=" * 50)
        print("CODING THEORY PROJECT MENU")
        print("=" * 50)
        print("Select an experiment to run:")
        print("1. Hamming code")
        print("2. Convolutional code")
        print("3. Concatenated code (Hamming + Convolutional)")
        print("4. All experiments")
        print("0. Exit")
        print("=" * 50)

        choice = input("Enter your choice: ")

        if choice == "1":
            hamming_only()

        elif choice == "2":
            convolutional_only()

        elif choice == "3":
            concatenated_only()

        elif choice == "4":
            run_all_experiments()

        elif choice == "0":
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 0.")

        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()