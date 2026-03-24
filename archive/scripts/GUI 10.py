import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QFileDialog, QListWidget
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QFileDialog, QListWidget, QSlider, QLabel, QHBoxLayout
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QListWidget, QSlider, QLabel,QButtonGroup

from PySide2.QtCore import Qt


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np  

USVtrain_path = '/Users/mikilon/Data/USVtrain'
LABELS = ["USV", "Quiet", "Noise", "NA"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Data Labeler")
        # Initialize variables for n_fft and hop_length with their default values
        self.n_fft = 512
        self.hop_length = 256
        self.layout = QVBoxLayout()

        # File list widget
        self.fileListWidget = QListWidget()
        self.layout.addWidget(self.fileListWidget)

        # Load button
        self.loadButton = QPushButton("Load Audio Files")
        self.loadButton.clicked.connect(self.loadFiles)
        self.layout.addWidget(self.loadButton)

        # n_fft slider setup        
        #Array of powers of 2 for n_fft
        self.powers_of_2 = 2**np.arange(7, 12)  # Adjust range as needed
        
        # Sliders layout
        sliders_layout = QHBoxLayout()

        # n_fft slider setup
        n_fft_label = QLabel("n_fft:")
        self.n_fft_slider = QSlider(Qt.Horizontal)
        self.n_fft_slider.setMinimum(0)  # Start index of powers_of_2 array
        self.n_fft_slider.setMaximum(len(self.powers_of_2) - 1)  # End index
        self.n_fft_slider.setValue(np.log2(self.n_fft) - 7)  # Convert default n_fft to index
        self.n_fft_slider.setTickInterval(1)
        self.n_fft_slider.setTickPosition(QSlider.TicksBelow)
        self.n_fft_slider.valueChanged.connect(self.update_n_fft)
        sliders_layout.addWidget(n_fft_label)
        sliders_layout.addWidget(self.n_fft_slider)
        
        # hop_length slider setup
        hop_length_label = QLabel("overlap:")
        self.hop_length_slider = QSlider(Qt.Horizontal)
        self.hop_length_slider.setMinimum(0)
        self.hop_length_slider.setMaximum(len(self.powers_of_2) - 1)  # End index
        self.hop_length_slider.setValue(np.log2(self.hop_length) - 7)   # Default value
        self.hop_length_slider.setTickInterval(1)
        self.hop_length_slider.setTickPosition(QSlider.TicksBelow)
        self.hop_length_slider.valueChanged.connect(self.update_hop_length)
        sliders_layout.addWidget(hop_length_label)
        sliders_layout.addWidget(self.hop_length_slider)
        
        self.layout.addLayout(sliders_layout)
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Labels layout
        labels_layout = QHBoxLayout()
        # Initialize button group for labels
        self.label_buttons_group = QButtonGroup()
        self.label_buttons_group.buttonClicked[int].connect(self.label_selection)  # Connect to slot with button ID
        
        # Dynamically create a button for each label
        for index, label in enumerate(LABELS):
            button = QPushButton(label)
            labels_layout.addWidget(button)
            self.label_buttons_group.addButton(button, index)  # Assign ID as index
        
        self.layout.addLayout(labels_layout)
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # This list will store the file paths of selected audio files
        self.selectedFiles = []
        # Initialize variables for selection start and end times
        self.selection_start = None
        self.selection_end = None

        # Variables to store selected region and label
        self.selected_region = None
        self.selected_label_index = None

    def label_selection(self, button_id):
        print(f"Label '{LABELS[button_id]}' was selected for the region.")
        # Here, you can associate the selected region with the label index (button_id)
        # This might involve storing the selection or updating the display to reflect the label
        if self.selected_region:
            self.selected_label_index = button_id
            # Further processing, e.g., storing the label with the region coordinates
            # You might also want to update the display to reflect the label selection
    
    def onselect(self, eclick, erelease):
        # eclick and erelease are the press and release events
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"Selection: ({x1}, {y1}) to ({x2}, {y2})")
        # Process the selection, e.g., display it, store it, etc.
        self.selection_start = x1
        self.selection_end = x2
        self.update_selection()

    def update_selection(self):
        if self.selection_start is not None and self.selection_end is not None:
            # Ensure selection_start is less than selection_end
            start, end = sorted([self.selection_start, self.selection_end])
            
            # Clear existing selection, if any
            self.figure.clear()
            
            # Redraw the spectrogram
            self.updateFigure()            
            # Draw the selection rectangle
            self.figure.axes[1].axvspan(start, end, color='red', alpha=0.5)
            
            self.canvas.draw()

    def loadFiles(self):
        default_path = USVtrain_path  # Set your default path here
        fileNames, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files", default_path, "Audio Files (*.wav *.mp3)")
        self.fileListWidget.clear()  # Clear existing entries
        self.fileListWidget.addItems(fileNames)  # Add new entries to the list widget


        # Update the selectedFiles list with new file paths
        self.selectedFiles = fileNames

        # Example: Process the first selected file (if any)
        if self.selectedFiles:
            self.processFile(self.selectedFiles[0])

    def load_audio_file(self, file_path):
        self.y, self.sr = librosa.load(file_path, sr=None)
    
    def processFile(self, filePath):
        # Placeholder for processing a file
        self.load_audio_file(filePath)
        D = librosa.stft(self.y, n_fft=self.n_fft , hop_length=self.hop_length)
        self.spectrogram_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        #print(spectrogram_db.min(), spectrogram_db.max())

        # Here you could load the audio data, generate a plot, etc.
        print(f"Processing file: {filePath}")
        self.setWindowTitle("Audio Data Labeler: " + filePath)
        self.updateFigure()
    
    def updateFigure(self):
        # Clear the current figure
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        # use librosa to display the variable y as a waveform
        librosa.display.waveshow(self.y, sr=self.sr, ax=ax1)
        ax1.set_yticks([])  # Remove y-axis ticks
        ax1.set_ylabel('')  # Remove y-axis label
        ax1.set_xticks([])  # Remove x-axis ticks for the top plot
        ax1.set_xlabel('')  # Remove x-axis label
        ax2 = self.figure.add_subplot(212, sharex=ax1)
        img = librosa.display.specshow(self.spectrogram_db, sr=self.sr, x_axis='time', y_axis='linear',hop_length=self.hop_length, ax=ax2, vmin=-60, vmax=0)
        ax2.set_yticks([])  # Remove y-axis ticks
        ax2.set_ylabel('')  # Remove y-axis label
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(True)
        ax2.set_xlabel('')  # Remove x-axis label
        # Initialize RectangleSelector
        self.toggle_selector = RectangleSelector(ax2, self.onselect, useblit=True,
                                                 button=[1],  # Left mouse button
                                                 minspanx=5, minspany=5,
                                                 spancoords='pixels',
                                                 interactive=True)

        
        
        self.canvas.draw()

    def update_n_fft(self, index):
        self.n_fft = self.powers_of_2[index]
        print(f"Updated n_fft: {self.n_fft}")
        self.updateFigure()
        # Call a method to redraw the spectrogram with the new n_fft value
    
    def update_hop_length(self, index):
        self.hop_length = self.powers_of_2[index]
        print(f"Updated hop_length: {self.hop_length}")
        self.updateFigure()
    
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
