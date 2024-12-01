import tkinter as tk
from tkinter import ttk
import pyaudio
import numpy as np
import threading
import queue
import whisper
from PIL import Image, ImageTk
import sounddevice as sd
import torch
from tkinter import messagebox
import time

class WhisperLiveUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech To Text")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Initialize Whisper
        self.model_size = "base"
        self.model = whisper.load_model(self.model_size)
        self.language = "English"
        
        self.setup_ui()
        self.setup_audio()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Speech To Text", 
                              font=('Arial', 24, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Audio level visualization
        self.canvas = tk.Canvas(main_frame, height=50, bg='#2d2d2d',
                              highlightthickness=0)
        self.canvas.pack(fill=tk.X, pady=(0, 20))
        self.audio_bar = self.canvas.create_rectangle(0, 0, 0, 50, 
                                                    fill='#007acc')
        
        # Service label
        service_label = ttk.Label(main_frame, text="Service: Whisper Live",
                                font=('Arial', 12))
        service_label.pack(pady=(0, 10))
        
        # Transcription display
        self.transcription_var = tk.StringVar()
        self.transcription_var.set("Speak something...")
        self.transcription_label = ttk.Label(main_frame, 
                                           textvariable=self.transcription_var,
                                           font=('Arial', 14))
        self.transcription_label.pack(pady=(0, 20))
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)
        
        # Language selection
        self.language_var = tk.StringVar(value="English")
        languages = ["English", "Hindi", "Spanish", "French", "German"]
        language_menu = ttk.Combobox(controls_frame, 
                                   textvariable=self.language_var,
                                   values=languages, state="readonly")
        language_menu.pack(side=tk.LEFT, padx=5)
        
        # Settings button
        self.setup_settings_button(controls_frame)
        
        # Start/Stop button
        self.toggle_button = ttk.Button(main_frame, text="Start Recording",
                                      command=self.toggle_recording)
        self.toggle_button.pack(pady=(20, 0))
        
    def setup_settings_button(self, parent):
        settings_frame = ttk.Frame(parent)
        settings_frame.pack(side=tk.RIGHT, padx=5)
        
        settings_button = ttk.Button(settings_frame, text="⚙️ Configure",
                                   command=self.show_settings)
        settings_button.pack()
        
    def show_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Configure Whisper Live")
        settings_window.geometry("400x300")
        
        # Model selection
        ttk.Label(settings_window, text="Model Size").pack(pady=10)
        model_sizes = ["tiny", "base", "small", "medium", "large"]
        model_var = tk.StringVar(value=self.model_size)
        model_menu = ttk.Combobox(settings_window, textvariable=model_var,
                                values=model_sizes, state="readonly")
        model_menu.pack()
        
        def apply_settings():
            new_model_size = model_var.get()
            if new_model_size != self.model_size:
                try:
                    self.model = whisper.load_model(new_model_size)
                    self.model_size = new_model_size
                    messagebox.showinfo("Success", 
                                      f"Switched to {new_model_size} model")
                except Exception as e:
                    messagebox.showerror("Error", 
                                       f"Failed to load model: {str(e)}")
            settings_window.destroy()
        
        ttk.Button(settings_window, text="Apply", 
                  command=apply_settings).pack(pady=20)
        
    def setup_audio(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
            self.toggle_button.config(text="Stop Recording")
        else:
            self.stop_recording()
            self.toggle_button.config(text="Start Recording")
    
    def start_recording(self):
        self.is_recording = True
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start visualization update
        self.update_visualization()
    
    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
            # Update visualization
            level = np.abs(audio_data).mean()
            self.root.after(0, self.update_audio_bar, level)
        return (in_data, pyaudio.paContinue)
    
    def update_audio_bar(self, level):
        # Scale the level to the canvas width
        width = self.canvas.winfo_width()
        bar_width = min(width * level * 50, width)  # Adjust multiplier as needed
        self.canvas.coords(self.audio_bar, 0, 0, bar_width, 50)
    
    def update_visualization(self):
        if self.is_recording:
            self.root.after(50, self.update_visualization)
    
    def process_audio(self):
        audio_data = []
        silence_threshold = 0.01
        silence_time = 0
        
        while self.is_recording:
            try:
                data = self.audio_queue.get(timeout=1)
                audio_data.append(data)
                
                # Check for silence
                if np.abs(data).mean() < silence_threshold:
                    silence_time += len(data) / self.RATE
                else:
                    silence_time = 0
                
                # Process after silence or accumulated audio
                if silence_time > 0.5 or len(audio_data) * self.CHUNK / self.RATE > 5:
                    if len(audio_data) > 0:
                        # Convert to audio array
                        audio_array = np.concatenate(audio_data)
                        
                        # Transcribe with Whisper
                        try:
                            result = self.model.transcribe(
                                audio_array,
                                language=self.language_var.get().lower(),
                                fp16=torch.cuda.is_available()
                            )
                            
                            # Update UI
                            if result["text"].strip():
                                self.root.after(0, self.update_transcription, 
                                              result["text"])
                        except Exception as e:
                            print(f"Transcription error: {str(e)}")
                        
                        # Clear buffer
                        audio_data = []
                        silence_time = 0
            
            except queue.Empty:
                continue
    
    def update_transcription(self, text):
        self.transcription_var.set(text)
    
    def __del__(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperLiveUI(root)
    root.mainloop()