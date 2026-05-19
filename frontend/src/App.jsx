import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:5000/predict";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile = (event) => {
    const selectedFile = event.target.files?.[0];

    setFile(selectedFile || null);
    setResult(null);

    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    } else {
      setPreview("");
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please select an image");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    try {
      setLoading(true);
      setResult(null);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setResult(data);
    } catch (error) {
      console.error(error);
      alert(error.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Skin Disease Detection</h1>

      <input type="file" accept="image/*" onChange={handleFile} />

      {preview && (
        <img src={preview} alt="Selected skin sample preview" className="preview" />
      )}

      <button type="button" onClick={handleSubmit} disabled={loading}>
        {loading ? "Predicting..." : "Predict Disease"}
      </button>

      {result && (
        <div className="result">
          <h2>Disease: {result.disease}</h2>
          <h3>Confidence: {result.confidence}</h3>
        </div>
      )}
    </div>
  );
}

export default App;
