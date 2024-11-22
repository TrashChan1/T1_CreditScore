def main_menu():
    while True:
        print("\nMain Menu")
        print("(1) Load data")
        print("(2) Process data")
        print("(3) Model details")
        print("(4) Test model")
        print("(5) Quit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            print("\nLoading and cleaning input data set")
            print("***********************************")
        elif choice == '2':
            print("\nProcessing input data set")
            print("*************************")
        elif choice == '3':
            print("\nPrinting model details")
            print("**********************")
        elif choice == '4':
            print("\nTesting model")
            print("*************")
        elif choice == '5':
            print("\nQuiting program, goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()

