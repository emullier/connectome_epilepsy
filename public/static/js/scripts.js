// declare globally the images to look for
const images = [
    { id: "biasImageConcat", src: "static/images/BiasPCA_concat.png", title: "PCA dimensionality reduction on concatenated data" , section:"bias"},
    { id: "biasImageProc1", src: "/static/images/BiasPCA_proc1.png", title: "PCA dimensionality reduction on processing 1", section:"bias" },
    { id: "consProc1", src: "/static/images/consensus_proc1.png", title: "Consensus- processing 1", section: "consensus" },
    { id: "consProc2", src: "/static/images/consensus_proc2.png", title: "Consensus- processing 2", section: "consensus" },
    { id: "reconstConcat", src: "/static/images/reconstruct_SC_procConcatenated.png", title: "Reconstructed SC - Concatenated", section: "harmonics" },
    { id: "reconstProc1", src: "/static/images/reconstruct_SC_proc1.png", title: "Reconstructed SC - processing 1", section: "harmonics" },
    { id: "reconstProc2", src: "/static/images/reconstruct_SC_proc2.png", title: "Reconstructed SC - processing 2", section: "harmonics" },
    { id: "procruProc1", src: "/static/images/RotationProcrustes_proc1.png", title: "Procrustes Alignement - processing 1", section: "harmonics" },
    { id: "procruProc2", src: "/static/images/RotationProcrustes_proc2.png", title: "Procrustes Alignement - processing 2", section: "harmonics" }
];


document.addEventListener("DOMContentLoaded", function() {
    // declare fetch and save config
    fetchConfig();

    // Load images for all sections
    const sections = ["bias", "consensus", "harmonics"];
    sections.forEach(section => {
        loadImages(section);
    });

    const saveButton = document.getElementById('saveConfigButton');
    if (saveButton) {
        saveButton.addEventListener('click', saveConfig);
    }
});


// Utility function to load images based on the active tab
function loadImages(section) {
    // Get the section div
    const sectionDiv = document.getElementById(section);
    if (!sectionDiv) {
        console.error(`Section div with ID ${section} not found.`);
        return;
    }
    // Clear previous images
    sectionDiv.innerHTML = '';
    // Filter images by the current section
    const filteredImages = images.filter(image => image.section === section);
    // Loop through the filtered images and load them
    filteredImages.forEach(function(image) {
        console.log(`Processing image: ${image.src}`);
        const titleElement = document.createElement("h3");
        titleElement.textContent = image.title;
        const imgElement = document.createElement("img");
        imgElement.id = image.id;
        imgElement.alt = "Not Processed";
        imgElement.style = "max-width: 100%; height: auto;";
        sectionDiv.appendChild(titleElement);
        sectionDiv.appendChild(imgElement);

        checkImageExists(image.src, function(exists, url) {
            if (exists) {
                imgElement.src = url;
            } else {
                imgElement.style.display = 'none';  // Hide the image if it no longer exists
                //titleElement.style.display = 'none';  // Optionally hide the title
            }
        });
    });
}
 
// Function to open the selected tab
function openTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active-button');
    });
    // Show the selected tab content
    document.getElementById(tabId).classList.add('active');
    // Set the active tab button
    document.querySelector(`.tab-button[onclick="openTab('${tabId}')"]`).classList.add('active-button');
    // Check if tabId is not 'config', then load images
    if (tabId !== 'config') {
        loadImages(tabId);
    }
}

// Function to fetch and populate the config from the server
async function fetchConfig() {
    try {
        const response = await fetch('/get-config');
        const config = await response.json();
        // Populate form with config values
        document.getElementById('scale').value = config.Parameters.scale;
        document.getElementById('metric').value = config.Parameters.metric;
        document.getElementById('data_dir').value = config.Parameters.data_dir;
        document.getElementById('info_path').value = config.Parameters.info_path;
        document.getElementById('output_dir').value = config.Parameters.output_dir;
        document.getElementById('fig_dir').value = config.Parameters.fig_dir;
        document.getElementById('load_dataset').checked = config.Parameters.load_dataset;
        document.getElementById('filters').value = config.Parameters.filters.join(',');
        document.getElementById('age_min').value = config.Parameters.age[0];
        document.getElementById('age_max').value = config.Parameters.age[1];
        document.getElementById('gender').value = config.Parameters.gender.join(',');
        document.getElementById('group').value = config.Parameters.group.join(',');
        document.getElementById('dwi').value = config.Parameters.dwi.join(',');
        document.getElementById('processing').value = Object.entries(config.Parameters.processing)
            .map(([k, v]) => `${k}:${v}`).join(',');
        document.getElementById('perform_bias_analysis').checked = config.BIAS.perform_analysis;
        document.getElementById('DimReduc').value = config.BIAS.DimReduc;
        document.getElementById('colors_key').value = config.BIAS.colors_key.join(',');
        document.getElementById('perform_consensus_analysis').checked = config.CONSENSUS.perform_analysis;
        document.getElementById('nbins').value = config.CONSENSUS.nbins;
        document.getElementById('generate_harmonics').checked = config.HARMONICS.generate_harmonics;
        document.getElementById('compare_harmonics').checked = config.HARMONICS.compare_harmonics;
        document.getElementById('consensus_harmonics').checked = config.HARMONICS.consensus;
        document.getElementById('perform_indvscons_analysis').checked = config.INDvsCONS.perform_analysis;
        document.getElementById('perform_sdi_analysis').checked = config.SDI.perform_analysis;
    } catch (error) {
        console.error('Error fetching config:', error);
    }
}

// Function to gather selected group values
function getSelectedGroups() {
    const selectedGroups = [];
    document.querySelectorAll('.group-checkbox:checked').forEach(checkbox => {
        selectedGroups.push(checkbox.value);
    });
    return selectedGroups;
}

// Function to save the config to the server
async function saveConfig() {
    const processingOptions = {
        shore: "cmp-3.1.0_shore",
        multitissue: "cmp-3.1.0_multitissue",
        L2008_DSI: "cmp-3.0-RC4_L2008_DSI",
        L2008_multish: "cmp-3.0-RC4_L2008_multishell",
        bmax5000: "cmp-3.1.0_bmax5000dsi",
        orig: "struct_data",
        example: "Individual_Connectomes"
    };

    // Get selected values from dropdowns
    const processing1 = document.getElementById('processing1').value;
    const processing2 = document.getElementById('processing2').value;

    const config = {
        Parameters: {
            scale: parseInt(document.getElementById('scale').value),
            metric: document.getElementById('metric').value,
            data_dir: document.getElementById('data_dir').value,
            info_path: document.getElementById('info_path').value,
            output_dir: document.getElementById('output_dir').value,
            fig_dir: document.getElementById('fig_dir').value,
            load_dataset: document.getElementById('load_dataset').checked,
            filters: document.getElementById('filters').value.split(','),
            age: [
                parseInt(document.getElementById('age_min').value),
                parseInt(document.getElementById('age_max').value)
            ],
            gender: document.getElementById('gender').value.split(','),
            group: getSelectedGroups(), // Use the function to get selected groups
            dwi: document.getElementById('dwi').value.split(','),
            //processing: Object.fromEntries(document.getElementById('processing').value.split(',')
            //    .map(item => item.split(':')))
            processing: {
                [processing1]: processingOptions[processing1],
                [processing2]: processingOptions[processing2]
            }
        },
        BIAS: {
            perform_analysis: document.getElementById('perform_bias_analysis').checked,
            DimReduc: document.getElementById('DimReduc').value,
            colors_key: document.getElementById('colors_key').value.split(',')
        },
        CONSENSUS: {
            perform_analysis: document.getElementById('perform_consensus_analysis').checked,
            nbins: parseInt(document.getElementById('nbins').value)
        },
        HARMONICS: {
            generate_harmonics: document.getElementById('generate_harmonics').checked,
            compare_harmonics: document.getElementById('compare_harmonics').checked,
            consensus: document.getElementById('consensus_harmonics').checked
        },
        INDvsCONS: {
            perform_analysis: document.getElementById('perform_indvscons_analysis').checked
        },
        SDI: {
            perform_analysis: document.getElementById('perform_sdi_analysis').checked
        }
    };

    try {
        const response = await fetch('/save-config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        if (response.ok) {
            alert('Configuration saved successfully');
        } else {
            alert('Error saving configuration');
        }
    } catch (error) {
        console.error('Error saving config:', error);
    }
}

// function to run python script
function runScript() {
    const outputDiv = document.getElementById('console-output');
    outputDiv.innerHTML = ''; // Clear previous output

    fetch('/run-script')
        .then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            function read() {
                reader.read().then(({ done, value }) => {
                    if (done) return;
                    const text = decoder.decode(value);
                    outputDiv.innerHTML += text;
                    outputDiv.scrollTop = outputDiv.scrollHeight; // Scroll to the bottom
                    read(); // Read next chunk
                }).catch(error => {
                    console.error('Error reading stream:', error);
                });
            }
            read(); // Start reading
        })
        .catch(error => console.error('Error running script:', error));
}


// function to check if an image exists
function checkImageExists(url, callback) {
    const img = new Image();
    img.onload = () => callback(true, url);
    img.onerror = () => callback(false, url);
    img.src = url;

    // Optional: Set a timeout to avoid long waits for missing images
    setTimeout(() => callback(false, url), 5000); // Timeout after 5 seconds
}





