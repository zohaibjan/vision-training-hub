export function createApi(token, fetcher = fetch) {
  async function request(path, options = {}) {
    const response = await fetcher('/api' + path, {
      ...options, headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json', ...options.headers },
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const detail = body.detail;
      throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
        ? detail.map(e => `${e.loc?.slice(1).join('.') || 'Request'}: ${e.msg}`).join('; ')
        : `Request failed (${response.status})`);
    }
    return response;
  }
  return {
    get: async (path, signal) => (await request(path, { signal })).json(),
    post: async (path, body) => (await request(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined })).json(),
    download: async (job, file) => (await request(`/jobs/${job}/artifacts/${file.split('/').map(encodeURIComponent).join('/')}`)).blob(),
  };
}
