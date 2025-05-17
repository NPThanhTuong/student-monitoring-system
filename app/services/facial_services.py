import os

from app.facial_recognition_services.FaceRecognitionSystem import FaceRecognitionSystem


def facialConsole():
    print("Face Recognition System with Database")
    print("====================================")

    face_system = FaceRecognitionSystem()

    while True:
        print("\nMenu:")
        print("1. Add face from webcam")
        print("2. Add face from image file")
        print("3. List all persons")
        print("4. Delete person")
        print("5. Start recognition")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            name = input("Enter name for the person: ")
            if name:
                success = face_system.add_face_from_webcam(name)
                if success:
                    print(f"Successfully added face for {name}")
                else:
                    print("Failed to add face")

        elif choice == '2':
            name = input("Enter name for the person: ")
            if name:
                image_path = input("Enter path to image file: ")
                if os.path.isfile(image_path):
                    success = face_system.add_face_from_image(name, image_path)
                    if success:
                        print(f"Successfully added face for {name} from {image_path}")
                    else:
                        print("Failed to add face from image")
                else:
                    print(f"File not found: {image_path}")

        elif choice == '3':
            face_system.list_all_persons()

        elif choice == '4':
            name = input("Enter name of person to delete: ")
            if name:
                confirmed = input(f"Are you sure you want to delete {name}? (y/n): ").lower() == 'y'
                if confirmed:
                    success = face_system.delete_person(name)
                    if success:
                        print(f"Successfully deleted {name}")
                    else:
                        print(f"Failed to delete {name}")

        elif choice == '5':
            face_system.recognize_from_webcam()

        elif choice == '6':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")