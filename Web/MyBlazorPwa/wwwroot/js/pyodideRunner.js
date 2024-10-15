let pyodide = null;

async function runPythonCode(pyCode) {
    if (!pyodide) {
        // Set the indexURL to point to the Pyodide distribution files
        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.18.1/full/"
        });

        // Load numpy
        await pyodide.loadPackage("numpy");
    }

    try {
        let result = await pyodide.runPythonAsync(pyCode);
        return result;
    } catch (error) {
        console.error("Python error:", error);
        return "Error: " + error;
    }
}
