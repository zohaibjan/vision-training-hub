// Read-only illustration of interface states. No real jobs, GPUs or model results.
const spec = { dataset: 'synthetic-boxes', model: 'yolov8n.pt', epochs: 50, batch: 8, imgsz: 640, seed: 42 };
export const previewJobs = [
  { id: 'sample-queued', status: 'queued', created: 1792292400, spec: { ...spec, name: 'Higher resolution experiment' }, progress: {}, tracking: {}, artifacts: [], gpu: null },
  { id: 'sample-running', status: 'running', created: 1792292100, spec: { ...spec, name: 'Object detection · baseline' }, progress: { epoch: 18, epochs: 50, simulated: true, metrics: {} }, tracking: {}, artifacts: [], gpu: 'demo-0' },
  { id: 'sample-completed', status: 'completed', created: 1792288800, spec: { ...spec, name: 'Dataset pipeline · smoke test', epochs: 2 }, progress: { epoch: 2, epochs: 2, simulated: true, metrics: {} }, tracking: {}, artifacts: [], gpu: 'demo-1' },
];
export const previewSystem = { mode: 'preview', worker_alive: false, worker: { gpus: [
  { uuid: 'demo-0', name: 'Sample GPU 01', free_mb: 8192, available: true },
  { uuid: 'demo-1', name: 'Sample GPU 02', free_mb: 8192, available: true },
] } };
