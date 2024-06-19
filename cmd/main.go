package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

func main() {

	http.Handle("/", http.FileServer(http.Dir("./public")))

	// Handle requests for generating JSON data
	http.HandleFunc("/generate-json", HandleGenerateJSONAndCallPythonScript)
	http.HandleFunc("/upload", uploadHandler)

	// Handle requests to the root route

	fmt.Println("Server started at :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("Error starting server:", err)
	}
}

// AudioData represents the structure of the audio data
type AudioData struct {
	ID          int       `json:"id"`
	FileName    string    `json:"file_name"`
	SR          int       `json:"sr"`
	ArrayLength int       `json:"arrayLength"`
	AudioArray  []float64 `json:"audio_array"`
	IsVoice     []bool    `json:"is_voice"`
	IsViolin    []bool    `json:"is_violin"`
	IsMridangam []bool    `json:"is_mridangam"`
	IsGhatam    []bool    `json:"is_ghatam"`
}

func HandleGenerateJSONAndCallPythonScript(w http.ResponseWriter, r *http.Request) {
	// Generate JSON file
	fileName := "input.json"
	err := GenerateAndWriteJSON(fileName)
	if err != nil {
		http.Error(w, "Error generating JSON: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Call Python script
	cmd := exec.Command("python3", "lib/script.py", fileName)
	err = cmd.Run() // Run the command without capturing output
	if err != nil {
		http.Error(w, "Error calling Python script: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Send a success status code
	w.WriteHeader(http.StatusOK)

	response := fmt.Sprintf(`
		<div id="output-content" class="text-center">
			<a href="./output.html" target="_blank" rel="noopener noreferrer">
				<img id="output-image" src="./output.png?t=%d" alt="Intervals" class="img-fluid">
			</a>
			<img id="sound-wave" src="./sound_wave.png?t=%d" alt="sound_wave" class="img-fluid">
			<div class="center-button">
				<a id="download-btn" href="./output.png" download="output.png" class="btn btn-secondary mt-3">Download Image</a>
			</div>
		</div>`, time.Now().Unix(), time.Now().Unix())

	fmt.Fprint(w, response)
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	// Parse the multipart form, limiting the size to 20 MB
	err := r.ParseMultipartForm(50 << 20)
	if err != nil {
		http.Error(w, "Failed to parse form", http.StatusBadRequest)
		fmt.Println("Error parsing form:", err)
		return
	}

	// Get the uploaded file
	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Failed to retrieve file", http.StatusBadRequest)
		fmt.Println("Error retrieving file:", err)
		return
	}
	defer file.Close()

	// Get the storage path from an environment variable
	storagePath := os.Getenv("STORAGE_PATH")
	if storagePath == "" {
		storagePath = "./" // default to current directory
	}

	// Create the destination file
	dst, err := os.Create(filepath.Join(storagePath, "inputAudio.mp3"))
	if err != nil {
		http.Error(w, "Failed to create file", http.StatusInternalServerError)
		fmt.Println("Error creating file:", err)
		return
	}
	defer dst.Close()

	// Copy the uploaded file to the destination
	if _, err := io.Copy(dst, file); err != nil {
		http.Error(w, "Failed to save file", http.StatusInternalServerError)
		fmt.Println("Error saving file:", err)
		return
	}
	fmt.Fprintf(w, "File uploaded successfully: inputAudio.mp3")

	cmd := exec.Command("python3", "lib/script2.py", "inputAudio.mp3")
	err = cmd.Run() // Run the command without capturing output
	if err != nil {
		http.Error(w, "Error calling Python script: "+err.Error(), http.StatusInternalServerError)
		return
	}

	response := fmt.Sprintf(`
		<div id="output-content" class="text-center">
			<div class="audio-container">
					<audio controls>
						<source src=../inputAudio.mp3" type="audio/mpeg">
						Your browser does not support the audio element.
					</audio>
            </div>
			<a href="./output.html" target="_blank" rel="noopener noreferrer">
				<img id="output-image" src="./output.png?t=%d" alt="Intervals" class="img-fluid">
			</a>
			<img id="sound-wave" src="./sound_wave.png?t=%d" alt="sound_wave" class="img-fluid">
			<div class="center-button">
				<div class="center-button">
                <a id="download-btn" href="./output.png" download="output.png" class="btn btn-secondary mt-3">Download Graph</a>
                <a id="download-btn" href="./output.html" download="output.html" class="btn btn-secondary mt-3">Download Interactive Graph</a>
                <a id="download-btn" href="./sound_wave.png" download="sound-wave.png" class="btn btn-secondary mt-3">Download Sound wave</a>
            </div>
			</div>
		</div>`, time.Now().Unix(), time.Now().Unix())

	fmt.Fprint(w, response)
}

// generateRandomBoolArray generates an array of random boolean values
func generateRandomBoolArray(length int) []bool {
	rand.NewSource(time.Now().UnixNano())
	boolArray := make([]bool, length)
	for i := range boolArray {
		boolArray[i] = rand.Intn(2) == 1
	}
	return boolArray
}

// generateRandomFloatArray generates an array of random float64 values
func generateRandomFloatArray(length int) []float64 {
	rand.NewSource(time.Now().UnixNano())
	floatArray := make([]float64, length)
	for i := range floatArray {
		floatArray[i] = rand.Float64()
	}
	return floatArray
}

// GenerateAndWriteJSON generates the audio data with random values and writes it to a JSON file
func GenerateAndWriteJSON(fileName string) error {
	// Define the directory for storing data files
	directory := "data"

	// Ensure the directory exists
	if err := os.MkdirAll(directory, 0755); err != nil {
		return fmt.Errorf("error creating directory: %v", err)
	}

	// Define the file path for the JSON file
	filePath := filepath.Join(directory, fileName)

	// Create the output file in the specified directory
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("error creating file: %v", err)
	}
	defer file.Close()

	// Define the length of the arrays
	arrayLength := 1000

	// Create an instance of AudioData with random values
	audioData := AudioData{
		ID:          0,
		FileName:    "performance1mixed.wav",
		SR:          44100,
		ArrayLength: arrayLength,
		AudioArray:  generateRandomFloatArray(arrayLength),
		IsVoice:     generateRandomBoolArray(arrayLength),
		IsViolin:    generateRandomBoolArray(arrayLength),
		IsMridangam: generateRandomBoolArray(arrayLength),
		IsGhatam:    generateRandomBoolArray(arrayLength),
	}

	// Convert the AudioData instance to JSON
	jsonData, err := json.MarshalIndent(audioData, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshalling to JSON: %v", err)
	}

	// Write the JSON data to the file
	_, err = file.Write(jsonData)
	if err != nil {
		return fmt.Errorf("error writing to file: %v", err)
	}

	return nil
}

func HandleGenerateJSON(w http.ResponseWriter, r *http.Request) {
	fileName := "input.json"
	err := GenerateAndWriteJSON(fileName)
	if err != nil {
		http.Error(w, "Error generating JSON: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("JSON file successfully created"))
}
