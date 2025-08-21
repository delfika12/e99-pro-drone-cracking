from DroneController import Drone

# Membuat instance drone
drone = Drone()

# Fungsi untuk menerjemahkan perintah manusia
def interpret_command(command):
    command = command.lower().strip()  # Mengubah input ke lowercase dan menghilangkan spasi

    # Menangani perintah yang sesuai
    if command == 'connect':
        drone.connect()
        drone.calibrate()
        print("Drone connected & calibrate.")
    elif command == 'calibrate':
        drone.calibrate()
        print("Drone calibrated.")
    elif command == 'takeoff':
        drone.take_off(2)  # Take off selama 3 detik
        print("Drone taking off...")
    elif command == 'land':
        drone.land(2)  # Land selama 2 detik
        print("Drone landing...")
    elif command == 'forward':
        drone.move_forward(30, 0.2)  # Maju dengan kecepatan 30% selama 0.2 detik
        print("Drone moving forward...")
    elif command == 'backward':
        drone.move_backward(30, 0.2)  # Mundur dengan kecepatan 30% selama 0.2 detik
        print("Drone moving backward...")
    elif command == 'left':
        drone.move_left(30, 0.2)  # Kiri dengan kecepatan 30% selama 0.2 detik
        print("Drone moving left...")
    elif command == 'right':
        drone.move_right(30, 0.2)  # Kanan dengan kecepatan 30% selama 0.2 detik
        print("Drone moving right...")
    elif command == 'up':
        drone.move_up(30, 0.2)  # Naik dengan kecepatan 30% selama 0.2 detik
        print("Drone moving up...")
    elif command == 'down':
        drone.move_down(30, 0.2)  # Turun dengan kecepatan 30% selama 0.2 detik
        print("Drone moving down...")
    elif command == 'stop':
        drone.stop()
        print("Drone stopped.")
    elif command == 'rotate left':
        drone.rotate_left(30, 0.2)  # Putar kiri dengan kecepatan 30% selama 0.2 detik
        print("Drone rotating left...")
    elif command == 'rotate right':
        drone.rotate_right(30, 0.2)  # Putar kanan dengan kecepatan 30% selama 0.2 detik
        print("Drone rotating right...")
    elif command == 'disconnect':
        drone.disconnect()
        print("Drone disconnected.")
    else:
        print("Perintah tidak dikenali, coba lagi!")

# Fungsi utama untuk menjalankan perintah dari user
def main():
    print("Selamat datang di kontrol drone!")
    print("Masukkan perintah (contoh: connect, calibrate, takeoff, forward, backward, etc.)")
    while True:
        user_input = input("Masukkan perintah: ")  # Menerima input dari pengguna
        if user_input == 'exit':
            print("Keluar dari kontrol drone.")
            break  # Keluar dari loop
        interpret_command(user_input)  # Menafsirkan dan menjalankan perintah

if __name__ == "__main__":
    main()
