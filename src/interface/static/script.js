// Utility to compute multiclass ROC/AUC
function computeRocAuc(yTrue, proba, stages) {
	// For each class compute FPR, TPR and AUC
	function auc(fpr, tpr) {
		let area = 0;
		for (let i = 1; i < fpr.length; i++) {
			area += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2;
		}
		return area;
	}
	const results = [];
	for (let c = 0; c < stages.length; c++) {
		// Binarise labels for one-vs-rest
		const yBin = yTrue.map(y => y === c ? 1 : 0);
		const scores = proba.map(row => row[c]);
		// Sort by score descending
		const sorted = scores.map((s, i) => [s, yBin[i]]).sort((a, b) => b[0] - a[0]);
		let tp = 0, fp = 0, fn = yBin.reduce((a, b) => a + b, 0), tn = yBin.length - fn;
		const tpr = [0], fpr = [0];
		for (let i = 0; i < sorted.length; i++) {
			if (sorted[i][1] === 1) { tp++; fn--; } else { fp++; tn--; }
			tpr.push(tp / (tp + fn));
			fpr.push(fp / (fp + tn));
		}
		results.push({
			label: stages[c],
			fpr,
			tpr,
			auc: auc(fpr, tpr)
		});
	}
	return results;
}

// Render probability histogram
function renderProbaHistogram(yTrue, proba, stages) {
	const ctx = document.getElementById('probaHistogramChart');
	if (!ctx) return;
	// Collect predicted probabilities for each class
	const data = stages.map((stage, i) => proba.map(row => row[i]));
	const bins = Array.from({length: 10}, (_, i) => i/10);
	const datasets = data.map((probs, i) => {
		const hist = Array(10).fill(0);
		probs.forEach(p => {
			const idx = Math.min(9, Math.floor(p*10));
			hist[idx]++;
		});
		return {
			label: stages[i],
			data: hist,
			backgroundColor: `rgba(${50+30*i},${100+20*i},235,0.5)`
		};
	});
	if (window.probaHistogramChart) window.probaHistogramChart.destroy();
	window.probaHistogramChart = new Chart(ctx, {
		type: 'bar',
		data: {
			labels: bins.map(b => `${(b*100).toFixed(0)}-${(b*100+10).toFixed(0)}%`),
			datasets: datasets
		},
		options: {
			responsive: true,
			plugins: { legend: { position: 'top' }, title: { display: true, text: 'Probability Distribution by Class' } },
			scales: { x: { stacked: true }, y: { stacked: true } }
		}
	});
}

// Render ROC/AUC curve
function renderRocCurve(yTrue, proba, stages) {
	const ctx = document.getElementById('rocCurveChart');
	const aucDiv = document.getElementById('aucScores');
	if (!ctx || !aucDiv) return;
	const results = computeRocAuc(yTrue, proba, stages);
	const datasets = results.map((res, i) => ({
		label: `${res.label} (AUC=${res.auc.toFixed(3)})`,
		data: res.fpr.map((fpr, j) => ({x: fpr, y: res.tpr[j]})),
		fill: false,
		borderColor: `hsl(${i*360/stages.length},70%,50%)`,
		tension: 0.1
	}));
	if (window.rocCurveChart) window.rocCurveChart.destroy();
	window.rocCurveChart = new Chart(ctx, {
		type: 'line',
		data: { datasets },
		options: {
			responsive: true,
			plugins: { legend: { position: 'top' }, title: { display: true, text: 'ROC Curve by Class' } },
			scales: { x: { min: 0, max: 1, title: { display: true, text: 'FPR' } }, y: { min: 0, max: 1, title: { display: true, text: 'TPR' } } }
		}
	});
	aucDiv.innerHTML = results.map(res => `<span class="badge bg-info me-1">${res.label}: AUC=${res.auc.toFixed(3)}</span>`).join(' ');
}

// Populate model selector for advanced charts
function renderProbaModelSelect(models, onChange) {
	const select = document.getElementById('probaModelSelect');
	if (!select) return;
	select.innerHTML = '';
	models.forEach(m => {
		const opt = document.createElement('option');
		opt.value = m.id;
		opt.textContent = m.name;
		select.appendChild(opt);
	});
	select.onchange = e => onChange(e.target.value);
}
function renderRocModelSelect(models, onChange) {
	const select = document.getElementById('rocModelSelect');
	if (!select) return;
	select.innerHTML = '';
	models.forEach(m => {
		const opt = document.createElement('option');
		opt.value = m.id;
		opt.textContent = m.name;
		select.appendChild(opt);
	});
	select.onchange = e => onChange(e.target.value);
}

// Fetch probabilities and render charts
async function fetchAndRenderProbaAndRoc(modelId) {
	const targetDataset = currentDataset;
	const probaResp = await fetch(`/api/probabilities?model=${modelId}&dataset=${targetDataset}`);
	if (!probaResp.ok) return;
	const payload = await probaResp.json();
	if (currentDataset !== targetDataset) return;
	if (payload.dataset_label) {
		updateDatasetLabels(payload.dataset_label);
	}
	const { y_true, proba, stages } = payload;
	renderProbaHistogram(y_true, proba, stages);
	renderRocCurve(y_true, proba, stages);
}
// Render visual ranking of models
function renderModelRanking(models) {
	const container = document.getElementById('model-ranking');
	if (!container) return;
	// Identify best values
	const best = {
		accuracy: Math.max(...models.map(m => m.metrics.accuracy)),
		macro_f1: Math.max(...models.map(m => m.metrics.macro_f1)),
		balanced_accuracy: Math.max(...models.map(m => m.metrics.balanced_accuracy)),
		loss: Math.min(...models.map(m => m.metrics.loss)),
	};
	let html = '<table class="table table-sm table-bordered align-middle text-center mb-0"><thead><tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Balanced Acc.</th><th>Log Loss</th></tr></thead><tbody>';
	models.forEach(m => {
		html += `<tr>`;
		html += `<td><b>${m.name}</b></td>`;
		html += `<td${m.metrics.accuracy === best.accuracy ? ' class="table-success fw-bold"' : ''}>${m.metrics.accuracy.toFixed(3)}${m.metrics.accuracy === best.accuracy ? ' <i class="bi bi-trophy-fill text-warning"></i>' : ''}</td>`;
		html += `<td${m.metrics.macro_f1 === best.macro_f1 ? ' class="table-success fw-bold"' : ''}>${m.metrics.macro_f1.toFixed(3)}${m.metrics.macro_f1 === best.macro_f1 ? ' <i class="bi bi-trophy-fill text-warning"></i>' : ''}</td>`;
		html += `<td${m.metrics.balanced_accuracy === best.balanced_accuracy ? ' class="table-success fw-bold"' : ''}>${m.metrics.balanced_accuracy.toFixed(3)}${m.metrics.balanced_accuracy === best.balanced_accuracy ? ' <i class="bi bi-trophy-fill text-warning"></i>' : ''}</td>`;
		html += `<td${m.metrics.loss === best.loss ? ' class="table-success fw-bold"' : ''}>${m.metrics.loss.toFixed(3)}${m.metrics.loss === best.loss ? ' <i class="bi bi-trophy-fill text-warning"></i>' : ''}</td>`;
		html += `</tr>`;
	});
	html += '</tbody></table>';
	container.innerHTML = html;
}

// Render detailed classification report table
function renderClassificationReport(models, stages, selectedModelId) {
	const container = document.getElementById('classification-report-table');
	if (!container) return;
	const model = models.find(m => m.id === selectedModelId) || models[0];
	if (!model || !model.classification_report) return;
	let html = '<table class="table table-sm table-bordered align-middle text-center mb-0"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead><tbody>';
	stages.forEach(stage => {
		const stats = model.classification_report[stage];
		html += `<tr><td>${stage}</td>`;
		html += `<td>${(stats?.precision ?? 0).toFixed(3)}</td>`;
		html += `<td>${(stats?.recall ?? 0).toFixed(3)}</td>`;
		html += `<td>${(stats?.['f1-score'] ?? 0).toFixed(3)}</td>`;
		html += `<td>${stats?.support ?? 0}</td></tr>`;
	});
	html += '</tbody></table>';
	container.innerHTML = html;
}

// Populate model selector for the classification report
function renderClassificationModelSelect(models, onChange) {
	const select = document.getElementById('classificationModelSelect');
	if (!select) return;
	select.innerHTML = '';
	models.forEach(m => {
		const opt = document.createElement('option');
		opt.value = m.id;
		opt.textContent = m.name;
		select.appendChild(opt);
	});
	select.onchange = e => onChange(e.target.value);
}
	// Update KPI cards with metrics from the selected model
	function renderKPI(models, selectedModelId) {
		const model = models.find(m => m.id === selectedModelId) || models[0];
		if (!model) return;
		document.getElementById('kpi-accuracy-value').textContent = model.metrics.accuracy.toFixed(3);
		document.getElementById('kpi-f1-value').textContent = model.metrics.macro_f1.toFixed(3);
		document.getElementById('kpi-balanced-value').textContent = model.metrics.balanced_accuracy.toFixed(3);
		document.getElementById('kpi-loss-value').textContent = model.metrics.loss.toFixed(3);
	}
	const confusionModelSelect = document.getElementById('confusionModelSelect');
	const confusionMatrixChartCanvas = document.getElementById('confusionMatrixChart');
	let confusionMatrixChart = null;
	// Render confusion matrix as a styled table
	function renderConfusionMatrix(models, stages, selectedModelId) {
		const tableContainer = document.getElementById('confusionMatrixTable');
		if (!tableContainer) return;
		const model = models.find(m => m.id === selectedModelId) || models[0];
		if (!model || !model.confusion_matrix) return;
		const matrix = model.confusion_matrix;
		// Determine the highest value to normalise cell colours
		const maxVal = Math.max(...matrix.flat());
		let html = '<table class="table table-bordered text-center align-middle confusion-matrix-table"><thead><tr><th></th>';
		stages.forEach(stage => { html += `<th class="bg-light">${stage}</th>`; });
		html += '</tr></thead><tbody>';
		matrix.forEach((row, i) => {
			html += `<tr><th class="bg-light">${stages[i]}</th>`;
			row.forEach((val, j) => {
				const intensity = maxVal ? Math.round(255 - 120 * (val / maxVal)) : 255;
				const bgColor = `background-color: rgb(${intensity},${intensity},255);`;
				html += `<td style="${bgColor}"><b>${val}</b></td>`;
			});
			html += '</tr>';
		});
		html += '</tbody></table>';
		tableContainer.innerHTML = html;
	}
	// Populate model selector for the confusion matrix
	function renderConfusionModelSelect(models, onChange) {
		if (!confusionModelSelect) return;
		confusionModelSelect.innerHTML = '';
		models.forEach(m => {
			const opt = document.createElement('option');
			opt.value = m.id;
			opt.textContent = m.name;
			confusionModelSelect.appendChild(opt);
		});
		confusionModelSelect.onchange = e => onChange(e.target.value);
	}
// Dashboard JS: renders metrics, simulation and controls

	document.addEventListener('DOMContentLoaded', function() {
	const metricsContainer = document.getElementById('metrics');
	const metricsTableContainer = document.getElementById('metrics-table');
	const metricsBarChartCanvas = document.getElementById('metricsBarChart');
	const controlsContainer = document.getElementById('controls');
	const logEl = document.getElementById('log');
	const datasetTabButtons = document.querySelectorAll('[data-dataset-button]');
	const datasetLabelElements = document.querySelectorAll('[data-dataset-label]');
	const datasetNames = { cassette: 'Sleep Cassette', telemetry: 'Sleep Telemetry' };
	let currentDataset = 'cassette';
	const activeDatasetButton = document.querySelector('[data-dataset-button].active');
	if (activeDatasetButton && datasetNames[activeDatasetButton.dataset.dataset]) {
		currentDataset = activeDatasetButton.dataset.dataset;
	}
	let simulationHandle = null;
	let metricsBarChart = null;
	let playButton = null;
	let stopButton = null;

	function updateDatasetLabels(label) {
		datasetLabelElements.forEach(el => {
			el.textContent = label;
		});
	}

	function setDataset(dataset) {
		if (!datasetNames[dataset] || dataset === currentDataset) {
			return;
		}
		currentDataset = dataset;
		datasetTabButtons.forEach(btn => {
			btn.classList.toggle('active', btn.dataset.dataset === currentDataset);
		});
		updateDatasetLabels(datasetNames[currentDataset]);
		stopSimulation();
		logEl.innerHTML = '';
		if (metricsBarChart) {
			metricsBarChart.destroy();
			metricsBarChart = null;
		}
		if (metricsTableContainer) {
			metricsTableContainer.innerHTML = '<div class="text-center text-muted py-3">Loading...</div>';
		}
		if (controlsContainer) {
			controlsContainer.innerHTML = '';
		}
		fetchModels();
	}

	function renderMetrics(models) {
		// Render table
		let html = '<table class="table table-striped table-bordered align-middle"><thead><tr><th>Model</th><th>Accuracy</th><th>Balanced Acc.</th><th>Macro F1</th><th>Log Loss</th></tr></thead><tbody>';
		models.forEach(model => {
			html += `<tr><td><b>${model.name}</b></td><td>${model.metrics.accuracy.toFixed(3)}</td>`;
			html += `<td>${model.metrics.balanced_accuracy.toFixed(3)}</td>`;
			html += `<td>${model.metrics.macro_f1.toFixed(3)}</td>`;
			html += `<td>${model.metrics.loss.toFixed(3)}</td></tr>`;
		});
		html += '</tbody></table>';
		if (metricsTableContainer) metricsTableContainer.innerHTML = html;

		// Render bar chart
		if (metricsBarChartCanvas) {
			const labels = models.map(m => m.name);
			const accuracy = models.map(m => m.metrics.accuracy);
			const macroF1 = models.map(m => m.metrics.macro_f1);
			const balancedAcc = models.map(m => m.metrics.balanced_accuracy);
			const loss = models.map(m => m.metrics.loss);
			const data = {
				labels: labels,
				datasets: [
					{
						label: 'Accuracy',
						data: accuracy,
						backgroundColor: 'rgba(31, 111, 235, 0.7)'
					},
					{
						label: 'Macro F1',
						data: macroF1,
						backgroundColor: 'rgba(76, 175, 80, 0.7)'
					},
					{
						label: 'Balanced Acc.',
						data: balancedAcc,
						backgroundColor: 'rgba(255, 193, 7, 0.7)'
					},
					{
						label: 'Log Loss',
						data: loss,
						backgroundColor: 'rgba(244, 67, 54, 0.7)'
					}
				]
			};
			const options = {
				responsive: true,
				plugins: {
					legend: { position: 'top' },
					title: { display: true, text: 'Model Metrics Comparison' }
				},
				scales: {
					y: {
						beginAtZero: true,
						ticks: { precision: 2 }
					}
				}
			};
			if (metricsBarChart) metricsBarChart.destroy();
			metricsBarChart = new Chart(metricsBarChartCanvas, {
				type: 'bar',
				data: data,
				options: options
			});
		}
	}

	function setSimulationRunning(running) {
		if (playButton) playButton.disabled = running;
		if (stopButton) stopButton.disabled = !running;
	}

	function setupControls() {
		controlsContainer.innerHTML = '';
		playButton = document.createElement('button');
		playButton.className = 'btn btn-primary me-2';
		playButton.textContent = 'Start full simulation';
		playButton.onclick = () => startSimulation();
		stopButton = document.createElement('button');
		stopButton.className = 'btn btn-outline-secondary';
		stopButton.textContent = 'Stop';
		stopButton.disabled = true;
		stopButton.onclick = () => {
			const label = datasetNames[currentDataset] || currentDataset;
			stopSimulation(`<span class="badge bg-secondary text-dark">Simulation stopped by the user (${label}).</span>`);
		};
		controlsContainer.appendChild(playButton);
		controlsContainer.appendChild(stopButton);
	}

			async function fetchModels() {
				const targetDataset = currentDataset;
				const label = datasetNames[targetDataset] || targetDataset;
				if (metricsBarChart) {
					metricsBarChart.destroy();
					metricsBarChart = null;
				}
				if (metricsTableContainer) {
					metricsTableContainer.innerHTML = `<div class="text-center text-muted py-3">Loading models for ${label}...</div>`;
				}
				if (controlsContainer) {
					controlsContainer.innerHTML = '';
				}
				let response;
				try {
					response = await fetch(`/api/models?dataset=${targetDataset}`);
				} catch (error) {
					if (metricsTableContainer) {
						metricsTableContainer.innerHTML = '<div class="alert alert-danger">Failed to reach the models endpoint.</div>';
					}
					return;
				}
				if (currentDataset !== targetDataset) {
					return;
				}
				if (response.status === 202) {
					if (metricsTableContainer) {
						metricsTableContainer.innerHTML = `<div class="text-center text-muted py-3">Loading models for ${label}...</div>`;
					}
					setTimeout(fetchModels, 2000);
					return;
				}
				if (!response.ok) {
					if (metricsTableContainer) {
						metricsTableContainer.innerHTML = `<div class="alert alert-danger">Failed to load models for ${label} (status ${response.status}).</div>`;
					}
					return;
				}
				const payload = await response.json();
				if (currentDataset !== targetDataset) {
					return;
				}
				const resolvedLabel = payload.dataset_label || label;
				updateDatasetLabels(resolvedLabel);
				renderMetrics(payload.models);
				setupControls();
				// KPIs and confusion matrix
				let selectedModelId = payload.models[0]?.id;
				renderKPI(payload.models, selectedModelId);
				renderConfusionModelSelect(payload.models, (modelId) => {
					renderConfusionMatrix(payload.models, payload.stages, modelId);
					renderKPI(payload.models, modelId);
				});
				renderConfusionMatrix(payload.models, payload.stages, selectedModelId);
				// Visual ranking of models
				renderModelRanking(payload.models);
				// Detailed classification report
				renderClassificationModelSelect(payload.models, (modelId) => {
					renderClassificationReport(payload.models, payload.stages, modelId);
				});
				renderClassificationReport(payload.models, payload.stages, selectedModelId);

				// Advanced charts: histograms and ROC/AUC
				renderProbaModelSelect(payload.models, (modelId) => {
					fetchAndRenderProbaAndRoc(modelId);
					renderRocModelSelect(payload.models, (id) => fetchAndRenderProbaAndRoc(id));
				});
				renderRocModelSelect(payload.models, (modelId) => fetchAndRenderProbaAndRoc(modelId));
				fetchAndRenderProbaAndRoc(selectedModelId);
			}

		function appendLog(message, cls = '', html = false) {
			const line = document.createElement('div');
			if (cls) line.classList.add(cls);
			if (html) {
				line.innerHTML = message;
			} else {
				line.textContent = message;
			}
			logEl.appendChild(line);
			logEl.scrollTop = logEl.scrollHeight;
		}

		async function startSimulation() {
			if (simulationHandle) {
				clearInterval(simulationHandle);
				simulationHandle = null;
			}
			setSimulationRunning(true);
			logEl.innerHTML = '';
			const targetDataset = currentDataset;
			const datasetLabel = datasetNames[targetDataset] || targetDataset;
			appendLog(`<span class="badge bg-primary">Starting full simulation (${datasetLabel})...</span>`, '', true);
			let response;
			try {
				response = await fetch(`/api/simulation?dataset=${targetDataset}`);
			} catch (error) {
				appendLog('<span class="badge bg-danger">Connection error while starting the simulation.</span>', 'incorrect', true);
				setSimulationRunning(false);
				return;
			}
			if (currentDataset !== targetDataset) {
				setSimulationRunning(false);
				return;
			}
			if (response.status === 202) {
				appendLog(`<span class="badge bg-warning text-dark">Models for ${datasetLabel} are still loading. Please wait a moment.</span>`, 'incorrect', true);
				setSimulationRunning(false);
				return;
			}
			if (!response.ok) {
				appendLog(`<span class="badge bg-danger">Failed to load simulation data for ${datasetLabel} (status ${response.status}).</span>`, 'incorrect', true);
				setSimulationRunning(false);
				return;
			}
			const payload = await response.json();
			if (currentDataset !== targetDataset) {
				setSimulationRunning(false);
				return;
			}
			const resolvedLabel = payload.dataset_label || datasetLabel;
			updateDatasetLabels(resolvedLabel);
			const frames = payload.frames;
			const models = payload.models;
			const simulationLabel = resolvedLabel;
			if (!Array.isArray(frames) || frames.length === 0) {
				appendLog(`<span class="badge bg-warning text-dark">No data available for the ${simulationLabel} simulation.</span>`, 'incorrect', true);
				setSimulationRunning(false);
				return;
			}
			let index = 0;
			simulationHandle = setInterval(() => {
				if (index >= frames.length) {
					appendLog(`<span class="badge bg-success">Simulation finished (${simulationLabel}).</span>`, '', true);
					clearInterval(simulationHandle);
					simulationHandle = null;
					setSimulationRunning(false);
					return;
				}
				const frame = frames[index];
				const parts = models.map(model => {
					const info = frame.predictions[model.id];
					const mark = info.correct
						? '<span class="badge bg-success ms-1"><i class="bi bi-check-circle"></i> OK</span>'
						: '<span class="badge bg-danger ms-1"><i class="bi bi-x-circle"></i> Error</span>';
					return `<span class="fw-bold">${model.name}</span>: <span class="text-info">${info.predicted}</span> ${mark}`;
				});
				const allCorrect = models.every(model => frame.predictions[model.id].correct);
				const status = allCorrect ? 'correct' : 'incorrect';
				appendLog(
					`<span class="me-2"><i class="bi bi-person-circle"></i> <b>${frame.subject_id}</b> <span class="text-secondary">(night ${frame.night_id})</span></span>` +
					`<span class="me-2"><i class="bi bi-clock-history"></i> <b>Epoch ${frame.epoch_idx}</b></span>` +
					`<span class="me-2"><i class="bi bi-flag"></i> <b>Actual:</b> <span class="text-warning">${frame.actual}</span></span>` +
					parts.join(' | '),
					status,
					true
				);
				index += 1;
			}, 200);
		}

		function stopSimulation(message = '') {
			if (simulationHandle) {
				clearInterval(simulationHandle);
				simulationHandle = null;
			}
			setSimulationRunning(false);
			if (message) {
				appendLog(message, '', true);
			}
		}

		datasetTabButtons.forEach(btn => {
			btn.addEventListener('click', () => {
				const targetDataset = btn.dataset.dataset;
				if (targetDataset) {
					setDataset(targetDataset);
				}
			});
		});

		updateDatasetLabels(datasetNames[currentDataset]);
		fetchModels();

		// Tab navigation code removed
});
