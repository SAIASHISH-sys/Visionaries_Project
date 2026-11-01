# Frontend

React 19 + TypeScript + Vite + Tailwind 4. The dev server proxies `/api` to the backend on port 8000.

    cd frontend
    npm install
    npm run dev

Start the backend first (see `backend/README.md`). Node is not on the PATH of this machine; a portable copy is in `.tools/node`.

## Screens

- **Workspace** (`/` and `/runs/:id`): load a DXF or DWG and view it, generate the three manifests, pick one for the compliance check, read the issues next to a plan with the failing rooms highlighted.
- **Compare** (`/runs/:id/compare`): the classical reading against GNN or raster, two linked plans, similarity metrics and a room-by-room table.

`PlanViewer` is the shared drawing component: it lays an SVG overlay (rooms, openings) over the server-rendered drawing, using the preview's `origin_mm` to convert manifest coordinates.
