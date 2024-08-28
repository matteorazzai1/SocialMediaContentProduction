import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, filedialog

from PIL import ImageTk

from AppLogic.captionGeneration import generateCaption
from AppLogic.imageGeneration import generateImage


class SocialMediaApp:
    def __init__(self,root):
        self.root = root
        self.root.title("Social Media Content Production")
        self.root.state('zoomed')

        self.style = ttk.Style()
        self.setup_styles()

        self.create_home_page()

    def submit_form(self,entry_name, entry_field):
        name = entry_name.get()
        field = entry_field.get()
        self.update_progress_bar(10)
        # Display the information or process it as needed
        caption = generateCaption(name,field)
        self.update_progress_bar(40)

        img = generateImage(self,caption,name,field)
        self.update_progress_bar(100)

        self.showPost(caption,img)

    def update_progress_bar(self, value):
        self.progress_bar['value'] = value
        self.root.update_idletasks()

    def setup_styles(self):
            soft_blue = "#8FAADC"
            dark_blue = "#1C3F94"
            white = "#FFFFFF"
            black = "#000000"

            self.style.configure('TFrame', background=soft_blue)
            self.style.configure('TLabel', background=soft_blue, foreground=white, font=('Helvetica', 14))
            self.style.configure('TButton', background=dark_blue, foreground=black, font=('Helvetica', 12), padding=10, relief="flat")
            self.style.map('TButton', background=[('active', dark_blue)], foreground=[('active', white)])

            self.root.configure(background=soft_blue)



    def create_home_page(self):
        # Create a frame for better control
        frame = ttk.Frame(self.root, padding="20")
        frame.place(relx=0.5, rely=0.5, anchor='center')

        # Configure rows and columns for better control
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # Create and place title label
        title_label = ttk.Label(frame, text="Social Media Content Production", font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(20, 10), sticky='n')

        # Create widgets and use grid layout
        label_name = ttk.Label(frame, text="Company name:")
        label_name.grid(row=1, column=0, padx=10, pady=10, sticky='e')

        entry_name = ttk.Entry(frame, width=30)
        entry_name.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        label_field = ttk.Label(frame, text="Field:")
        label_field.grid(row=2, column=0, padx=10, pady=10, sticky='e')

        entry_field = ttk.Entry(frame, width=30)
        entry_field.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        submit_button = ttk.Button(frame, text="Submit", command=lambda: self.submit_form(entry_name, entry_field))
        submit_button.grid(row=3, column=0, columnspan=3, padx=10, pady=20, sticky='n')

        # Create a progress bar
        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.grid(row=4, column=1, padx=10, pady=20, sticky='n')

        # Ensure the frame expands in the center
        self.root.update_idletasks()
        frame_width = frame.winfo_width()
        frame_height = frame.winfo_height()
        self.root.geometry(f"{frame_width + 40}x{frame_height + 100}")
        self.root.mainloop()

    def showPost(self, caption, img):
        # Create a new window
        new_window = tk.Toplevel(self.root)
        new_window.title("Generated Post")

        # Display the caption
        caption_label = tk.Label(new_window, text=caption)
        caption_label.pack()

        # Display the image
        img = ImageTk.PhotoImage(img)  # Convert to a format tkinter can display
        img_label = tk.Label(new_window, image=img)
        img_label.image = img  # Keep a reference to avoid garbage collection
        img_label.pack()


if __name__ == '__main__':
    root = tk.Tk()
    app = SocialMediaApp(root)
    root.mainloop()