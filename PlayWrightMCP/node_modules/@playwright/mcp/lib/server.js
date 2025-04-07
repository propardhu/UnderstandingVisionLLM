"use strict";
/**
 * Copyright (c) Microsoft Corporation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ServerList = void 0;
exports.createServerWithTools = createServerWithTools;
const index_js_1 = require("@modelcontextprotocol/sdk/server/index.js");
const types_js_1 = require("@modelcontextprotocol/sdk/types.js");
const context_1 = require("./context");
function createServerWithTools(options) {
    const { name, version, tools, resources } = options;
    const context = new context_1.Context(options);
    const server = new index_js_1.Server({ name, version }, {
        capabilities: {
            tools: {},
            resources: {},
        }
    });
    server.setRequestHandler(types_js_1.ListToolsRequestSchema, async () => {
        return { tools: tools.map(tool => tool.schema) };
    });
    server.setRequestHandler(types_js_1.ListResourcesRequestSchema, async () => {
        return { resources: resources.map(resource => resource.schema) };
    });
    server.setRequestHandler(types_js_1.CallToolRequestSchema, async (request) => {
        const tool = tools.find(tool => tool.schema.name === request.params.name);
        if (!tool) {
            return {
                content: [{ type: 'text', text: `Tool "${request.params.name}" not found` }],
                isError: true,
            };
        }
        try {
            const result = await tool.handle(context, request.params.arguments);
            return result;
        }
        catch (error) {
            return {
                content: [{ type: 'text', text: String(error) }],
                isError: true,
            };
        }
    });
    server.setRequestHandler(types_js_1.ReadResourceRequestSchema, async (request) => {
        const resource = resources.find(resource => resource.schema.uri === request.params.uri);
        if (!resource)
            return { contents: [] };
        const contents = await resource.read(context, request.params.uri);
        return { contents };
    });
    const oldClose = server.close.bind(server);
    server.close = async () => {
        await oldClose();
        await context.close();
    };
    return server;
}
class ServerList {
    _servers = [];
    _serverFactory;
    constructor(serverFactory) {
        this._serverFactory = serverFactory;
    }
    async create() {
        const server = this._serverFactory();
        this._servers.push(server);
        return server;
    }
    async close(server) {
        const index = this._servers.indexOf(server);
        if (index !== -1)
            this._servers.splice(index, 1);
        await server.close();
    }
    async closeAll() {
        await Promise.all(this._servers.map(server => server.close()));
    }
}
exports.ServerList = ServerList;
