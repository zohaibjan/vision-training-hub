# Hardware acceptance

Automated checks: 38 Python tests and 13 React tests, plus a Vite production build. Frontend tests use a mocked transport; Python integration tests run actual simulation subprocesses. Public preview browser checks cover rendering, filtering and log details. These checks do not substitute for a live CUDA/MLflow training run.

The automated suite validates control-plane behavior and a simulated subprocess runner. Before claiming a GPU deployment is validated:

1. Install matching NVIDIA drivers/PyTorch and record Python, torch, CUDA, ultralytics and mlflow versions (`pip freeze` plus `nvidia-smi`).
2. Start MLflow and run a two-epoch YOLOv8n job with the synthetic dataset, image size 64 and batch 1. Confirm CUDA execution in the log, completed status, downloadable `weights/best.pt` and `results.csv`.
3. Verify the displayed MLflow run ID exists, with epoch losses, validation metrics, parameters, dataset SHA and model artifacts. Synthetic accuracy is not a performance benchmark.
4. On a two-GPU machine, submit three jobs. Check two distinct UUIDs run concurrently and the third queues, then starts when a GPU is released.
5. Run an unrelated CUDA workload and verify its GPU is skipped. Test GPU discovery failure: no jobs should start.
6. Cancel a queued job and a long active job. Confirm the process group exits and the GPU reservation is reusable. Inspect MLflow status after graceful cancellation.
7. Restart the worker during training. Verify interrupted status, no duplicate training, preserved logs, and normal dispatch of pending jobs. Also test abrupt supervisor termination and lock retention.
8. Test a deliberate out-of-memory run and an unavailable MLflow endpoint. Confirm a visible failure and useful logs, followed by successful dispatch of another job.
9. Check portal layout and controls in desktop/mobile browsers and authenticated artifact downloads.

Do not expose MLflow or the portal directly to the public internet. Use your own non-sensitive data for this checklist.
