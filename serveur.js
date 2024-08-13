const express = require('express');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const app = express();
const PORT = 3000;

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Middleware to parse JSON bodies
app.use(express.json());

// Endpoint to get the configuration
app.get('/get-config', (req, res) => {
    fs.readFile('config.json', 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading config file:', err);
            return res.status(500).send('Error reading config file');
        }
        res.json(JSON.parse(data));
    });
});

// Endpoint to save the configuration
app.post('/save-config', (req, res) => {
    const config = req.body;
    fs.writeFile('config.json', JSON.stringify(config, null, 2), 'utf8', err => {
        if (err) {
            console.error('Error saving config file:', err);
            return res.status(500).send('Error saving config file');
        }
        res.send('Configuration saved successfully');
    });
});

// Endpoint to run the script
app.post('/run-script', (req, res) => {
    exec('python main.py', (error, stdout, stderr) => {
        if (error) {
            console.error('exec error:', error);
            console.error('stderr:', stderr);
            return res.status(500).send(`Error running script: ${stderr}`);
        }
        console.log('stdout:', stdout);
        // Assuming the script generates image.png in the public/images directory
        res.send(`Script executed successfully: ${stdout}`);
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
