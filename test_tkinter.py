import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Tkinter Test")
    root.geometry("300x200")
    label = tk.Label(root, text="Tkinter is working!")
    label.pack(pady=50)
    root.mainloop()

if __name__ == "__main__":
    main()
