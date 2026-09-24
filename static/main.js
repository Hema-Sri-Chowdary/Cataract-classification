document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const previewBox = document.getElementById('previewBox');
    const imgPreview = document.getElementById('imgPreview');
    const btnRemove = document.getElementById('btnRemove');
    const btnAnalyze = document.getElementById('btnAnalyze');
    
    const samplesScroll = document.getElementById('samplesScroll');
    
    const statsRow = document.getElementById('statsRow');
    const statPrediction = document.getElementById('statPrediction');
    const statConfidence = document.getElementById('statConfidence');
    const statCoverage = document.getElementById('statCoverage');
    
    const lowerGrid = document.getElementById('lowerGrid');
    
    // 5-Panel Images
    const p1Original = document.getElementById('p1Original');
    const p2Mask = document.getElementById('p2Mask');
    const p3Roi = document.getElementById('p3Roi');
    const p4Heatmap = document.getElementById('p4Heatmap');
    const p5Overlay = document.getElementById('p5Overlay');
    const p5Title = document.getElementById('p5Title');
    
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tab5panel = document.getElementById('tab5panel');
    const tabSingleView = document.getElementById('tabSingleView');
    const imgSingleView = document.getElementById('imgSingleView');
    
    const probRows = document.getElementById('probRows');
    const metricIoU = document.getElementById('metricIoU');
    const metricDice = document.getElementById('metricDice');
    const metricCoverage = document.getElementById('metricCoverage');

    let currentSelectedFile = null;
    let currentSelectedSamplePath = null;
    let lastAnalysisData = null;

    // Load dataset samples
    fetchSamples();

    async function fetchSamples() {
        try {
            const resp = await fetch('/api/samples');
            const data = await resp.json();
            if (data.status === 'success' && data.samples.length > 0) {
                renderSamples(data.samples);
            }
        } catch (e) {
            console.error('Error fetching samples:', e);
        }
    }

    function renderSamples(samples) {
        samplesScroll.innerHTML = '';
        samples.forEach((sample) => {
            const img = document.createElement('img');
            img.className = 'sample-thumb';
            img.src = sample.thumbnail;
            img.alt = sample.filename;
            img.addEventListener('click', () => selectSample(sample, img));
            samplesScroll.appendChild(img);
        });
    }

    function selectSample(sample, imgEl) {
        currentSelectedFile = null;
        currentSelectedSamplePath = sample.path;

        document.querySelectorAll('.sample-thumb').forEach(t => t.classList.remove('active'));
        imgEl.classList.add('active');

        imgPreview.src = sample.thumbnail;
        uploadPlaceholder.classList.add('hidden');
        previewBox.classList.remove('hidden');

        btnAnalyze.disabled = false;
    }

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        currentSelectedSamplePath = null;
        document.querySelectorAll('.sample-thumb').forEach(t => t.classList.remove('active'));

        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }

        currentSelectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imgPreview.src = e.target.result;
            uploadPlaceholder.classList.add('hidden');
            previewBox.classList.remove('hidden');
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        currentSelectedFile = null;
        currentSelectedSamplePath = null;
        fileInput.value = '';
        uploadPlaceholder.classList.remove('hidden');
        previewBox.classList.add('hidden');
        btnAnalyze.disabled = true;
        document.querySelectorAll('.sample-thumb').forEach(t => t.classList.remove('active'));
    });

    // Run Analysis
    btnAnalyze.addEventListener('click', runPipeline);

    async function runPipeline() {
        if (!currentSelectedFile && !currentSelectedSamplePath) return;

        btnAnalyze.disabled = true;
        btnAnalyze.textContent = 'Analyzing Image...';

        let response;
        try {
            if (currentSelectedFile) {
                const formData = new FormData();
                formData.append('image', currentSelectedFile);
                response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
            } else if (currentSelectedSamplePath) {
                response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sample_path: currentSelectedSamplePath })
                });
            }

            const data = await response.json();
            if (data.status === 'success') {
                lastAnalysisData = data.data;
                renderResults(data.data);
            } else {
                alert('Analysis error: ' + data.message);
            }
        } catch (e) {
            console.error(e);
            alert('Server error occurred during analysis.');
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.textContent = 'Analyze Image';
        }
    }

    function renderResults(res) {
        // Show Sections
        statsRow.classList.remove('hidden');
        lowerGrid.classList.remove('hidden');

        // Populate Stat Cards
        statPrediction.textContent = res.predicted_label;
        statConfidence.textContent = `${res.confidence}%`;
        statCoverage.textContent = `${res.metrics.coverage}%`;

        // Populate 5-Panel Images
        p1Original.src = res.visualizations.original;
        p2Mask.src = res.visualizations.mask;
        p3Roi.src = res.visualizations.roi;
        p4Heatmap.src = res.visualizations.heatmap;
        p5Overlay.src = res.visualizations.overlay;

        const gradeShortName = res.predicted_label.split(' (')[0];
        p5Title.textContent = `5. Overlay (${gradeShortName}: ${res.confidence}%)`;

        // Populate Class Probabilities
        probRows.innerHTML = '';
        const allowedClasses = ['Mild', 'Normal', 'Severe'];
        for (const [clsName, probVal] of Object.entries(res.probabilities)) {
            if (!allowedClasses.includes(clsName)) continue;
            const isActive = clsName === gradeShortName;
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <div class="prob-row-header ${isActive ? 'active' : ''}">
                    <span>${clsName}</span>
                    <strong>${probVal}%</strong>
                </div>
                <div class="prob-slider-bg">
                    <div class="prob-slider-fill" style="width: ${probVal}%;"></div>
                </div>
            `;
            probRows.appendChild(row);
        }

        // Populate Alignment Metrics
        metricIoU.textContent = res.metrics.iou.toFixed(4);
        metricDice.textContent = res.metrics.dice.toFixed(4);
        metricCoverage.textContent = `${res.metrics.coverage}%`;

        // Scroll to results smoothly
        statsRow.scrollIntoView({ behavior: 'smooth' });
    }

    // Tab Switcher
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const tabTarget = btn.getAttribute('data-tab');

            if (tabTarget === 'tab5panel') {
                tab5panel.classList.remove('hidden');
                tabSingleView.classList.add('hidden');
            } else if (lastAnalysisData) {
                tab5panel.classList.add('hidden');
                tabSingleView.classList.remove('hidden');

                if (tabTarget === 'tabOriginal') imgSingleView.src = lastAnalysisData.visualizations.original;
                if (tabTarget === 'tabSeg') imgSingleView.src = lastAnalysisData.visualizations.mask;
                if (tabTarget === 'tabRoi') imgSingleView.src = lastAnalysisData.visualizations.roi;
                if (tabTarget === 'tabGradcam') imgSingleView.src = lastAnalysisData.visualizations.overlay;
            }
        });
    });
});
