// import runMCP from '@playwright/mcp';

// console.log("🚀 MCP server starting at http://localhost:3000");

// runMCP({
//   port: 3000,
//   headless: false
// });
import { createServer } from '@playwright/mcp';

// ...

const server = createServer({
  launchOptions: { headless: false }
});
server.start;
console.log(server);
