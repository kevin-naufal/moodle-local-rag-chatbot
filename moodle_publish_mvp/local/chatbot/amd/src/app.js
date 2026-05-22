define(['core/log', 'local_chatbot/api_client'], function(Log, ApiClient) {
    const MAX_HISTORY = 80;

    const safeReadHistory = (key) => {
        try {
            const raw = window.localStorage.getItem(key);
            if (!raw) {
                return [];
            }
            const parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
        } catch (e) {
            return [];
        }
    };

    const safeWriteHistory = (key, value) => {
        try {
            window.localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            // Ignore storage failures.
        }
    };

    const normalizeMessageType = (type) => {
        const next = String(type || '').trim().toLowerCase();
        return next === 'user' ? 'user' : 'assistant';
    };

    const normalizeHistoryEntry = (entry) => {
        const source = (entry && typeof entry === 'object') ? entry : {};
        return {
            type: normalizeMessageType(source.type),
            text: String(source.text || ''),
            sources: Array.isArray(source.sources) ? source.sources.map((item) => String(item || '')) : [],
            requestId: String(source.requestId || ''),
            chatMode: String(source.chatMode || ''),
            questionId: String(source.questionId || ''),
            runId: Math.max(0, parseInt(source.runId, 10) || 0),
            questionText: String(source.questionText || ''),
            courseId: Math.max(0, parseInt(source.courseId, 10) || 0),
            topic: String(source.topic || '')
        };
    };

    const trimHistory = (history) => {
        const next = Array.isArray(history) ? history.map(normalizeHistoryEntry) : [];
        return next.slice(-MAX_HISTORY);
    };

    const userMessageCount = (history) => trimHistory(history).filter((entry) => entry.type === 'user').length;

    const createHistoryAdapter = (userid) => {
        const storageKey = `local_chatbot_history_u${userid || 'anon'}`;
        return {
            key: storageKey,
            read: () => trimHistory(safeReadHistory(storageKey)),
            write: (history) => safeWriteHistory(storageKey, trimHistory(history))
        };
    };

    const createInitialState = (config, historyAdapter) => {
        const history = historyAdapter.read();
        return {
            bootConfig: config,
            materialContext: {
                mode: 'none',
                isManual: false,
                disableTopicSelect: false,
                courseId: 0,
                topic: ''
            },
            materialsState: {
                activeFiles: [],
                selectedFile: '',
                parseStatus: {
                    status: 'no_materials',
                    is_parsed: false,
                    parsed_at: 0,
                    sources: 0
                },
                selectedEmbeddingStatus: null
            },
            chatState: {
                history: history,
                usageCount: userMessageCount(history),
                composerText: ''
            },
            previewState: {
                fileType: '',
                viewUrl: '',
                textContent: '',
                isLoading: false,
                error: ''
            },
            uiState: {
                statusText: String(config.statusnodocs || 'No materials selected'),
                isAppReady: false,
                mountMode: String(config.renderermode || 'legacy-php'),
                isChatBusy: false,
                isUploadBusy: false,
                isRefreshEmbeddingBusy: false,
                lastError: ''
            }
        };
    };

    const createStore = (initialState) => {
        let state = initialState;
        let listeners = [];

        const getState = () => state;

        const setState = (updater) => {
            const next = typeof updater === 'function'
                ? updater(state)
                : Object.assign({}, state, updater);
            state = next;
            listeners.slice().forEach((listener) => listener(state));
            return state;
        };

        const subscribe = (listener) => {
            listeners.push(listener);
            return () => {
                listeners = listeners.filter((item) => item !== listener);
            };
        };

        return {
            getState: getState,
            setState: setState,
            subscribe: subscribe
        };
    };

    const syncRootDataset = (root, state) => {
        root.dataset.appReady = state.uiState.isAppReady ? '1' : '0';
        root.dataset.mountMode = String(state.uiState.mountMode || 'legacy-php');
        root.dataset.materialMode = String(state.materialContext.mode || 'none');
        root.dataset.chatUsage = String(state.chatState.usageCount || 0);
        root.dataset.activeFileCount = String((state.materialsState.activeFiles || []).length);
    };

    const mountLegacyShell = (root, store) => {
        root.classList.add('local-chatbot-app-mounted');
        root.setAttribute('data-app-shell', 'legacy');
        syncRootDataset(root, store.getState());

        return store.subscribe((state) => {
            syncRootDataset(root, state);
        });
    };

    const createActions = (store, historyAdapter) => ({
        setStatusText: (statusText) => {
            store.setState((state) => Object.assign({}, state, {
                uiState: Object.assign({}, state.uiState, {
                    statusText: String(statusText || '')
                })
            }));
        },
        setComposerText: (composerText) => {
            store.setState((state) => Object.assign({}, state, {
                chatState: Object.assign({}, state.chatState, {
                    composerText: String(composerText || '')
                })
            }));
        },
        replaceHistory: (history) => {
            const normalizedHistory = trimHistory(history);
            historyAdapter.write(normalizedHistory);
            store.setState((state) => Object.assign({}, state, {
                chatState: Object.assign({}, state.chatState, {
                    history: normalizedHistory,
                    usageCount: userMessageCount(normalizedHistory)
                })
            }));
        }
    });

    return {
        init: function(config) {
            const bootConfig = (config && typeof config === 'object') ? config : {};
            const rootId = String(bootConfig.approotid || 'local-chatbot-app').trim();
            const root = document.getElementById(rootId);
            if (!root) {
                Log.debug('local_chatbot app root not found');
                return null;
            }

            const historyAdapter = createHistoryAdapter(bootConfig.userid || 'anon');
            const store = createStore(createInitialState(bootConfig, historyAdapter));
            const actions = createActions(store, historyAdapter);
            const api = ApiClient.create(bootConfig);

            const app = {
                root: root,
                api: api,
                store: store,
                actions: actions,
                historyAdapter: historyAdapter,
                config: bootConfig,
                destroy: null
            };

            const unsubscribe = mountLegacyShell(root, store);
            app.destroy = () => {
                unsubscribe();
                delete root.__localChatbotApp;
                if (window.localChatbotApp === app) {
                    delete window.localChatbotApp;
                }
            };

            root.__localChatbotApp = app;
            window.localChatbotApp = app;

            store.setState((state) => Object.assign({}, state, {
                uiState: Object.assign({}, state.uiState, {
                    isAppReady: true
                })
            }));

            Log.debug('local_chatbot app scaffold initialized');
            return app;
        }
    };
});
