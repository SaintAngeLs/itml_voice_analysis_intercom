let mediaRecorder;
let recordedChunks = [];

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        recordedChunks = [];
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        mediaRecorder.start();
    } catch (error) {
        console.error("Error accessing microphone: ", error);
        return "Error accessing microphone.";
    }

    return "Recording started.";
}

function stopRecording() {
    if (!mediaRecorder) {
        return "No recording in progress.";
    }

    mediaRecorder.stop();

    const audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });
    return window.URL.createObjectURL(audioBlob);  // Return the URL of the recorded audio
}
