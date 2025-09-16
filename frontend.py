import tkinter as tk
#create the main application window
root  = tk.Tk()
root.title("Simple Tkinter App")
root.geometry("200x100")

# Fuction to print "Hello, World!" in the console
 def say_hello():   # ✅ correct indentation
    print("Hello")

    print("good bye")

# Create a button that calls that triggers the say_hello function
hello_button = tk.Button(root, text="Click Me", command=say_hello)
hello_button.pack(pady=20) #pack the button into the window

#start the Tkinter event loop
root.mainloop()