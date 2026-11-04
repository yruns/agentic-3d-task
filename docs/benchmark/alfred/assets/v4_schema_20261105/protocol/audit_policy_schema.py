"""Local transparent Responses relay: record tool names only, never headers."""
import json
from pathlib import Path
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

root = Path('/home/ysh/codebase/agentic-3d-task/tmp/alfred_setup')
app = FastAPI()

@app.api_route('/{path:path}', methods=['GET', 'POST'])
async def relay(path: str, request: Request):
    body = await request.body()
    if path.endswith('responses'):
        payload = json.loads(body)
        record = {'model': payload.get('model'),
                  'tool_names': [item.get('name', item.get('type')) for item in payload.get('tools', [])],
                  'tool_count': len(payload.get('tools', [])),
                  'route': 'native Responses raw-byte relay', 'text_format':payload.get('text'), 'response_format':payload.get('response_format')}
        with (root / 'actor_tool_audit.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
    client = httpx.AsyncClient(timeout=300)
    headers = {key:value for key,value in request.headers.items() if key.lower() not in ('host','content-length','connection')}
    upstream = await client.send(client.build_request(request.method, 'http://127.0.0.1:8787/'+path,
                                                     content=body,headers=headers),stream=True)
    async def response_bytes():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
    response_headers = {key:value for key,value in upstream.headers.items()
                        if key.lower() not in ('content-length','transfer-encoding','connection')}
    return StreamingResponse(response_bytes(), status_code=upstream.status_code, headers=response_headers)

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8791,access_log=False)
