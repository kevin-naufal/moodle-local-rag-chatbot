define(['core/log', 'local_chatbot/api_client'], function(Log, ApiClient) {
    const MAX_HISTORY = 80;
    const MATERIALS_SYNC_EVENT = 'local-chatbot:materials-sync';

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
                ownsMaterialsPreview: Boolean(config.appownsmaterialspreview),
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

    const getDomRefs = () => ({
        statusWrap: document.getElementById('local-chatbot-status'),
        previewBody: document.getElementById('local-chatbot-preview-body'),
        previewName: document.getElementById('local-chatbot-preview-name'),
        previewEmbeddingStatus: document.getElementById('local-chatbot-preview-embedding-status'),
        refreshEmbeddingBtn: document.getElementById('local-chatbot-refresh-embedding-btn'),
        chatClassInput: document.getElementById('local-chatbot-chat-class'),
        chatTopicInput: document.getElementById('local-chatbot-chat-topic'),
        uploadInput: document.getElementById('local-chatbot-upload-input'),
        uploadBtn: document.getElementById('local-chatbot-upload-btn'),
        clearUploadBtn: document.getElementById('local-chatbot-clear-upload-btn'),
        materialContextWrap: document.getElementById('local-chatbot-material-context'),
        filesWrap: document.getElementById('local-chatbot-files')
    });

    const formatUnixTime = (value) => {
        const numeric = Math.max(0, parseInt(value, 10) || 0);
        if (!numeric) {
            return '-';
        }
        try {
            return new Date(numeric * 1000).toLocaleString();
        } catch (error) {
            return '-';
        }
    };

    const humanizeParseStatus = (value) => {
        const normalized = String(value || '').trim().toLowerCase();
        if (normalized === 'parsed') {
            return 'parsed';
        }
        if (normalized === 'needs_parsing') {
            return 'needs parsing';
        }
        if (normalized === 'no_materials') {
            return 'no materials';
        }
        return normalized || '-';
    };

    const getSelectedFile = (state) => String((state.materialsState && state.materialsState.selectedFile) || '').trim();

    const normalizeMaterialContext = (context) => {
        const next = (context && typeof context === 'object') ? context : {};
        return {
            mode: String(next.mode || 'none').trim().toLowerCase() || 'none',
            isManual: Boolean(next.is_manual),
            disableTopicSelect: Boolean(next.disable_topic_select),
            courseId: Math.max(0, parseInt(next.course_id, 10) || 0),
            topic: String(next.topic || '').trim()
        };
    };

    const createEmptyPreviewState = () => ({
        fileType: '',
        viewUrl: '',
        textContent: '',
        isLoading: false,
        error: ''
    });

    const dispatchMaterialsSync = (root, state) => {
        const detail = {
            materialContext: {
                mode: state.materialContext.mode,
                is_manual: state.materialContext.isManual,
                disable_topic_select: state.materialContext.disableTopicSelect,
                course_id: state.materialContext.courseId,
                topic: state.materialContext.topic
            },
            activeFiles: Array.isArray(state.materialsState.activeFiles) ? state.materialsState.activeFiles : [],
            selectedFile: getSelectedFile(state),
            parseStatus: state.materialsState.parseStatus || {
                status: 'no_materials',
                is_parsed: false,
                parsed_at: 0,
                sources: 0
            },
            selectedEmbeddingStatus: state.materialsState.selectedEmbeddingStatus || null,
            statusText: String(state.uiState.statusText || '')
        };
        root.dispatchEvent(new CustomEvent(MATERIALS_SYNC_EVENT, {
            bubbles: false,
            detail: detail
        }));
    };

    const syncRootDataset = (root, state) => {
        root.dataset.appReady = state.uiState.isAppReady ? '1' : '0';
        root.dataset.mountMode = String(state.uiState.mountMode || 'legacy-php');
        root.dataset.materialMode = String(state.materialContext.mode || 'none');
        root.dataset.chatUsage = String(state.chatState.usageCount || 0);
        root.dataset.activeFileCount = String((state.materialsState.activeFiles || []).length);
        root.dataset.ownsMaterialsPreview = state.uiState.ownsMaterialsPreview ? '1' : '0';
    };

    const populateTopicOptions = (chatTopicInput, config, courseTopics, courseValue, selectedTopic = '') => {
        if (!chatTopicInput) {
            return;
        }
        const topics = courseTopics[String(courseValue || '')] || [];
        chatTopicInput.innerHTML = '';

        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = config.coursetopicplaceholder || 'Select topic';
        chatTopicInput.appendChild(placeholder);

        topics.forEach((topic) => {
            const value = String((topic && topic.value) || (topic && topic.label) || '').trim();
            if (value === '') {
                return;
            }
            const option = document.createElement('option');
            option.value = value;
            option.textContent = String((topic && topic.label) || value);
            chatTopicInput.appendChild(option);
        });

        if (selectedTopic) {
            chatTopicInput.value = selectedTopic;
        }
    };

    const renderFiles = (filesWrap, files, nofiles, selectedFile, onFileClick) => {
        if (!filesWrap) {
            return;
        }

        if (!Array.isArray(files) || files.length === 0) {
            filesWrap.innerHTML = `<p class="local-chatbot-empty">${nofiles}</p>`;
            return;
        }

        filesWrap.innerHTML = '';
        files.forEach((file) => {
            const name = String((file && file.name) || '');
            if (!name) {
                return;
            }
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'local-chatbot-file-item';
            item.dataset.file = name;
            if (selectedFile === name) {
                item.classList.add('active');
            }
            item.innerHTML = `<span>${name}</span>`;
            item.addEventListener('click', () => onFileClick(name));
            filesWrap.appendChild(item);
        });
    };

    const renderPreviewEmbeddingStatus = (previewEmbeddingStatus, state, config) => {
        if (!previewEmbeddingStatus) {
            return;
        }

        const selectedFile = getSelectedFile(state);
        if (!selectedFile) {
            previewEmbeddingStatus.innerHTML = 'Select a file to view embedding status.';
            return;
        }

        const effectiveInfo = state.materialsState.selectedEmbeddingStatus && typeof state.materialsState.selectedEmbeddingStatus === 'object'
            ? state.materialsState.selectedEmbeddingStatus
            : null;
        const fileInActiveCorpus = effectiveInfo
            ? Boolean(effectiveInfo.file_in_active_corpus)
            : state.materialsState.activeFiles.some((file) => String((file && file.name) || '') === selectedFile);
        const isEmbedded = effectiveInfo
            ? Boolean(effectiveInfo.is_embedded_in_active_index)
            : (fileInActiveCorpus && Boolean(state.materialsState.parseStatus.is_parsed));
        const statusValue = effectiveInfo
            ? String(effectiveInfo.parse_status || '')
            : String((state.materialsState.parseStatus && state.materialsState.parseStatus.status) || '');
        const parsedAt = effectiveInfo
            ? formatUnixTime(effectiveInfo.parsed_at)
            : formatUnixTime(state.materialsState.parseStatus && state.materialsState.parseStatus.parsed_at);
        const embeddingModel = effectiveInfo && effectiveInfo.embedding_model
            ? String(effectiveInfo.embedding_model)
            : String(config.embedmodelbert || config.embedmodelollama || '-');

        previewEmbeddingStatus.innerHTML =
            `<div><strong>Selected file:</strong> ${selectedFile}</div>`
            + '<div><strong>Index scope:</strong> active corpus</div>'
            + `<div><strong>File in active corpus:</strong> ${fileInActiveCorpus ? 'yes' : 'no'}</div>`
            + `<div><strong>Embedded in current index:</strong> ${isEmbedded ? 'yes' : 'no'}</div>`
            + `<div><strong>Index status:</strong> ${humanizeParseStatus(statusValue)}</div>`
            + `<div><strong>Current embedding model:</strong> ${embeddingModel}</div>`
            + `<div><strong>Last indexed:</strong> ${parsedAt}</div>`;
    };

    const renderPreviewBody = (previewBody, state, config) => {
        if (!previewBody) {
            return;
        }

        const selectedFile = getSelectedFile(state);
        if (!selectedFile) {
            previewBody.innerHTML = `<p class="local-chatbot-empty">${config.previewempty || 'No preview available.'}</p>`;
            return;
        }

        if (state.previewState.isLoading) {
            previewBody.innerHTML = `<p class="local-chatbot-empty">${config.previewloading || 'Loading preview...'}</p>`;
            return;
        }

        if (state.previewState.error) {
            previewBody.innerHTML = `<p class="local-chatbot-empty">${state.previewState.error}</p>`;
            return;
        }

        if (state.previewState.fileType === 'pdf' && state.previewState.viewUrl) {
            previewBody.innerHTML = '';
            const frame = document.createElement('iframe');
            frame.src = state.previewState.viewUrl;
            frame.title = selectedFile;
            frame.loading = 'lazy';
            previewBody.appendChild(frame);

            const note = document.createElement('p');
            note.className = 'local-chatbot-preview-note';
            note.textContent = String(
                config.previewpdffallback || 'If the PDF does not render here, open it in a new tab.'
            );
            previewBody.appendChild(note);

            const link = document.createElement('a');
            link.href = state.previewState.viewUrl;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = String(config.previewopenpdf || 'Open PDF in new tab');
            previewBody.appendChild(link);
            return;
        }

        if (state.previewState.fileType === 'txt') {
            const pre = document.createElement('pre');
            pre.textContent = String(state.previewState.textContent || '');
            previewBody.innerHTML = '';
            previewBody.appendChild(pre);
            return;
        }

        previewBody.innerHTML = `<p class="local-chatbot-empty">${config.previewempty || 'No preview available.'}</p>`;
    };

    const renderMaterialsPreviewDomain = (app, refs) => {
        const state = app.store.getState();
        const config = app.config;
        const selectedFile = getSelectedFile(state);

        syncRootDataset(app.root, state);

        if (refs.statusWrap) {
            refs.statusWrap.textContent = String(state.uiState.statusText || '');
        }

        if (refs.chatClassInput) {
            refs.chatClassInput.disabled = Boolean(state.materialContext.disableTopicSelect || state.materialContext.isManual);
            if (state.materialContext.mode === 'topic' && state.materialContext.courseId > 0) {
                refs.chatClassInput.value = String(state.materialContext.courseId);
            } else if (state.materialContext.isManual) {
                refs.chatClassInput.value = '';
            }
        }

        if (refs.chatTopicInput) {
            const selectedCourse = refs.chatClassInput ? String(refs.chatClassInput.value || '').trim() : '';
            populateTopicOptions(
                refs.chatTopicInput,
                config,
                config.coursetopics || {},
                selectedCourse,
                state.materialContext.isManual ? '' : state.materialContext.topic
            );
            refs.chatTopicInput.disabled = Boolean(state.materialContext.disableTopicSelect || state.materialContext.isManual);
        }

        if (refs.clearUploadBtn) {
            refs.clearUploadBtn.disabled = !config.canmanualupload || !state.materialContext.isManual || state.uiState.isUploadBusy;
        }
        if (refs.uploadBtn) {
            refs.uploadBtn.disabled = !config.canmanualupload || state.uiState.isUploadBusy;
        }
        if (refs.refreshEmbeddingBtn) {
            refs.refreshEmbeddingBtn.disabled = !selectedFile || state.uiState.isRefreshEmbeddingBusy;
        }
        if (refs.materialContextWrap) {
            if (state.materialContext.isManual) {
                refs.materialContextWrap.textContent = String(
                    config.manualmodeactive || 'Manual upload is active. Clear uploaded materials to use class/topic materials again.'
                );
                refs.materialContextWrap.classList.remove('local-chatbot-hidden');
            } else {
                refs.materialContextWrap.textContent = '';
                refs.materialContextWrap.classList.add('local-chatbot-hidden');
            }
        }

        if (refs.previewName) {
            refs.previewName.textContent = selectedFile || '-';
        }

        renderFiles(
            refs.filesWrap,
            state.materialsState.activeFiles,
            config.nofiles || 'No active materials loaded.',
            selectedFile,
            (filename) => app.openFilePreview(filename)
        );
        renderPreviewEmbeddingStatus(refs.previewEmbeddingStatus, state, config);
        renderPreviewBody(refs.previewBody, state, config);
        dispatchMaterialsSync(app.root, state);
    };

    const createActions = (app, refs) => {
        const {store, historyAdapter, config} = app;

        const setStatusText = (statusText) => {
            store.setState((state) => Object.assign({}, state, {
                uiState: Object.assign({}, state.uiState, {
                    statusText: String(statusText || '')
                })
            }));
        };

        const setComposerText = (composerText) => {
            store.setState((state) => Object.assign({}, state, {
                chatState: Object.assign({}, state.chatState, {
                    composerText: String(composerText || '')
                })
            }));
        };

        const replaceHistory = (history) => {
            const normalizedHistory = trimHistory(history);
            historyAdapter.write(normalizedHistory);
            store.setState((state) => Object.assign({}, state, {
                chatState: Object.assign({}, state.chatState, {
                    history: normalizedHistory,
                    usageCount: userMessageCount(normalizedHistory)
                })
            }));
        };

        const applyMaterialPayload = (payload, options = {}) => {
            const activeFiles = Array.isArray(payload && payload.files) ? payload.files : [];
            const parseStatus = payload && payload.parse_status && typeof payload.parse_status === 'object'
                ? payload.parse_status
                : {
                    status: 'no_materials',
                    is_parsed: false,
                    parsed_at: 0,
                    sources: 0
                };
            const materialContext = normalizeMaterialContext(payload && payload.material_context);

            store.setState((state) => {
                const selectedFile = state.materialsState.selectedFile
                    && activeFiles.some((file) => String((file && file.name) || '') === state.materialsState.selectedFile)
                    ? state.materialsState.selectedFile
                    : '';
                const selectedEmbeddingStatus = selectedFile ? state.materialsState.selectedEmbeddingStatus : null;
                const nextPreviewState = selectedFile ? state.previewState : createEmptyPreviewState();
                const defaultStatus = !activeFiles.length
                    ? (config.statusnodocs || 'No materials selected')
                    : (materialContext.isManual
                        ? (config.manualmodeactive || 'Manual upload is active. Clear uploaded materials to use class/topic materials again.')
                        : (config.statusready || 'RAG ready'));

                return Object.assign({}, state, {
                    materialContext: materialContext,
                    materialsState: {
                        activeFiles: activeFiles,
                        selectedFile: selectedFile,
                        parseStatus: parseStatus,
                        selectedEmbeddingStatus: selectedEmbeddingStatus
                    },
                    previewState: nextPreviewState,
                    uiState: Object.assign({}, state.uiState, {
                        statusText: String(options.statusText || defaultStatus),
                        lastError: ''
                    })
                });
            });
        };

        const resetPreviewForSelection = (selectedFile, statusText, isLoading = false) => {
            store.setState((state) => Object.assign({}, state, {
                materialsState: Object.assign({}, state.materialsState, {
                    selectedFile: String(selectedFile || ''),
                    selectedEmbeddingStatus: null
                }),
                previewState: {
                    fileType: '',
                    viewUrl: '',
                    textContent: '',
                    isLoading: Boolean(isLoading),
                    error: ''
                },
                uiState: Object.assign({}, state.uiState, {
                    statusText: String(statusText || state.uiState.statusText || '')
                })
            }));
        };

        const setPreviewError = (message) => {
            store.setState((state) => Object.assign({}, state, {
                materialsState: Object.assign({}, state.materialsState, {
                    selectedEmbeddingStatus: null
                }),
                previewState: {
                    fileType: '',
                    viewUrl: '',
                    textContent: '',
                    isLoading: false,
                    error: String(message || config.previewerror || 'Failed to generate preview.')
                }
            }));
        };

        const setPreviewContent = (payload) => {
            store.setState((state) => Object.assign({}, state, {
                materialsState: Object.assign({}, state.materialsState, {
                    selectedEmbeddingStatus: payload.embedding_status || null
                }),
                previewState: {
                    fileType: String(payload.filetype || ''),
                    viewUrl: String(payload.viewurl || ''),
                    textContent: String(payload.content || ''),
                    isLoading: false,
                    error: ''
                }
            }));
        };

        const setMaterialBusy = (key, value) => {
            store.setState((state) => Object.assign({}, state, {
                uiState: Object.assign({}, state.uiState, {
                    [key]: Boolean(value)
                })
            }));
        };

        return {
            setStatusText: setStatusText,
            setComposerText: setComposerText,
            replaceHistory: replaceHistory,
            applyMaterialPayload: applyMaterialPayload,
            resetPreviewForSelection: resetPreviewForSelection,
            setPreviewError: setPreviewError,
            setPreviewContent: setPreviewContent,
            setMaterialBusy: setMaterialBusy
        };
    };

    const attachMaterialsPreviewDomain = (app, refs) => {
        if (!app.store.getState().uiState.ownsMaterialsPreview) {
            return () => {};
        }

        const {api, actions, config, store} = app;
        const unbinders = [];

        const loadActiveMaterials = async() => {
            actions.setStatusText(config.previewloading || 'Loading...');
            try {
                const payload = await api.listFiles();
                if (!payload || !payload.ok) {
                    throw new Error((payload && payload.error) || (config.chaterror || 'Failed to load materials.'));
                }
                actions.applyMaterialPayload(payload);
            } catch (error) {
                actions.applyMaterialPayload({
                    files: [],
                    material_context: {mode: 'none'},
                    parse_status: {
                        status: 'no_materials',
                        is_parsed: false,
                        parsed_at: 0,
                        sources: 0
                    }
                }, {
                    statusText: error && error.message ? error.message : (config.chaterror || 'Failed to process chat request.')
                });
                actions.setPreviewError(config.previewerror || 'Failed to generate preview.');
            }
        };

        const loadMaterialContext = async() => {
            const currentState = store.getState();
            if (currentState.materialContext.isManual) {
                return;
            }

            const courseid = refs.chatClassInput ? String(refs.chatClassInput.value || '').trim() : '';
            const topic = refs.chatTopicInput ? String(refs.chatTopicInput.value || '').trim() : '';

            actions.resetPreviewForSelection('', config.previewempty || 'No preview available.');

            if (!courseid || !topic) {
                actions.applyMaterialPayload({
                    files: [],
                    material_context: {
                        mode: 'none',
                        course_id: Number(courseid || 0),
                        topic: topic
                    },
                    parse_status: {
                        status: 'no_materials',
                        is_parsed: false,
                        parsed_at: 0,
                        sources: 0
                    }
                }, {
                    statusText: config.statusnodocs || 'No materials selected'
                });
                return;
            }

            actions.setStatusText(config.previewloading || 'Loading...');
            try {
                const payload = await api.setMaterialContext(courseid, topic);
                if (!payload || !payload.ok) {
                    throw new Error((payload && payload.error) || (config.chaterror || 'Failed to load materials.'));
                }
                actions.applyMaterialPayload(payload);
            } catch (error) {
                actions.applyMaterialPayload({
                    files: [],
                    material_context: {
                        mode: 'none',
                        course_id: Number(courseid || 0),
                        topic: topic
                    },
                    parse_status: {
                        status: 'no_materials',
                        is_parsed: false,
                        parsed_at: 0,
                        sources: 0
                    }
                }, {
                    statusText: error && error.message ? error.message : (config.chaterror || 'Failed to process chat request.')
                });
                actions.setPreviewError(config.previewerror || 'Failed to generate preview.');
            }
        };

        app.openFilePreview = async(filename) => {
            const selectedFile = String(filename || '').trim();
            if (!selectedFile) {
                return;
            }

            actions.resetPreviewForSelection(
                selectedFile,
                config.previewloading || 'Loading preview...',
                true
            );

            try {
                const payload = await api.fileContent(selectedFile);
                if (!payload || !payload.ok) {
                    throw new Error((payload && payload.error) || (config.previewerror || 'Failed to generate preview.'));
                }
                actions.setPreviewContent(payload);
            } catch (error) {
                actions.setPreviewError(
                    error && error.message ? error.message : (config.previewerror || 'Failed to generate preview.')
                );
            }
        };

        const uploadMaterials = async() => {
            if (!refs.uploadInput || !refs.uploadInput.files || !refs.uploadInput.files.length) {
                window.alert(config.manualuploadrequired || 'Choose at least one PDF or TXT file first.');
                return;
            }

            actions.setMaterialBusy('isUploadBusy', true);
            actions.setStatusText(config.manualuploading || 'Uploading materials...');

            try {
                const payload = await api.uploadMaterials(refs.uploadInput.files);
                if (!payload || !payload.ok) {
                    throw new Error((payload && payload.error) || (config.chaterror || 'Failed to process chat request.'));
                }
                if (refs.uploadInput) {
                    refs.uploadInput.value = '';
                }
                actions.applyMaterialPayload(payload);
            } catch (error) {
                actions.setStatusText(
                    error && error.message ? error.message : (config.chaterror || 'Failed to process chat request.')
                );
            } finally {
                actions.setMaterialBusy('isUploadBusy', false);
            }
        };

        const clearUploadedMaterials = async() => {
            const currentState = store.getState();
            if (!currentState.materialContext.isManual) {
                return;
            }

            actions.setMaterialBusy('isUploadBusy', true);
            try {
                const payload = await api.clearUploadedMaterials();
                if (!payload || !payload.ok) {
                    throw new Error((payload && payload.error) || (config.chaterror || 'Failed to process chat request.'));
                }
                actions.applyMaterialPayload(payload, {
                    statusText: config.manualcleared || 'Manual uploaded materials cleared. Topic selection is enabled again.'
                });
            } catch (error) {
                actions.setStatusText(
                    error && error.message ? error.message : (config.chaterror || 'Failed to process chat request.')
                );
            } finally {
                actions.setMaterialBusy('isUploadBusy', false);
            }
        };

        const refreshSelectedEmbedding = async() => {
            const state = store.getState();
            const fallbackFile = Array.isArray(state.materialsState.activeFiles) && state.materialsState.activeFiles.length
                ? String((state.materialsState.activeFiles[0] && state.materialsState.activeFiles[0].name) || '').trim()
                : '';
            const targetFile = getSelectedFile(state) || fallbackFile;
            if (!targetFile) {
                actions.setStatusText(config.refreshembeddingrequired || 'Load materials first.');
                return;
            }

            actions.setMaterialBusy('isRefreshEmbeddingBusy', true);
            actions.setStatusText(config.refreshembeddingloading || 'Refreshing embedding index...');

            try {
                const payload = await api.refreshSelectedEmbedding(targetFile);
                if (!payload || !payload.ok) {
                    throw new Error((payload && payload.error) || (config.refreshembeddingerror || 'Failed to refresh embedding index.'));
                }
                actions.applyMaterialPayload(payload, {
                    statusText:
                        `${config.refreshembeddingok || 'Embedding index refreshed for the active corpus.'} `
                        + `file=${payload.filename || targetFile}, sources=${payload.sources ?? '-'}`
                        + (payload.embedding_model ? `, embedding=${payload.embedding_model}` : '')
                });
                store.setState((currentState) => Object.assign({}, currentState, {
                    materialsState: Object.assign({}, currentState.materialsState, {
                        selectedEmbeddingStatus: payload.embedding_status || null
                    })
                }));
            } catch (error) {
                actions.setStatusText(
                    error && error.message ? error.message : (config.refreshembeddingerror || 'Failed to refresh embedding index.')
                );
            } finally {
                actions.setMaterialBusy('isRefreshEmbeddingBusy', false);
            }
        };

        if (refs.chatClassInput) {
            const onClassChange = () => {
                populateTopicOptions(
                    refs.chatTopicInput,
                    config,
                    config.coursetopics || {},
                    String(refs.chatClassInput.value || '').trim(),
                    ''
                );
                if (refs.chatTopicInput) {
                    refs.chatTopicInput.value = '';
                }
                loadMaterialContext();
            };
            refs.chatClassInput.addEventListener('change', onClassChange);
            unbinders.push(() => refs.chatClassInput.removeEventListener('change', onClassChange));
        }

        if (refs.chatTopicInput) {
            const onTopicChange = () => {
                loadMaterialContext();
            };
            refs.chatTopicInput.addEventListener('change', onTopicChange);
            unbinders.push(() => refs.chatTopicInput.removeEventListener('change', onTopicChange));
        }

        if (refs.uploadBtn) {
            const onUploadClick = () => {
                uploadMaterials();
            };
            refs.uploadBtn.addEventListener('click', onUploadClick);
            unbinders.push(() => refs.uploadBtn.removeEventListener('click', onUploadClick));
        }

        if (refs.clearUploadBtn) {
            const onClearClick = () => {
                clearUploadedMaterials();
            };
            refs.clearUploadBtn.addEventListener('click', onClearClick);
            unbinders.push(() => refs.clearUploadBtn.removeEventListener('click', onClearClick));
        }

        if (refs.refreshEmbeddingBtn) {
            const onRefreshClick = () => {
                refreshSelectedEmbedding();
            };
            refs.refreshEmbeddingBtn.addEventListener('click', onRefreshClick);
            unbinders.push(() => refs.refreshEmbeddingBtn.removeEventListener('click', onRefreshClick));
        }

        populateTopicOptions(
            refs.chatTopicInput,
            config,
            config.coursetopics || {},
            refs.chatClassInput ? String(refs.chatClassInput.value || '').trim() : '',
            ''
        );

        renderMaterialsPreviewDomain(app, refs);
        loadActiveMaterials();

        return () => {
            unbinders.forEach((unbind) => unbind());
        };
    };

    return {
        init: function(config) {
            const bootConfig = (config && typeof config === 'object') ? config : {};
            const rootId = String(bootConfig.approotid || 'local-chatbot-app').trim();
            const root = document.getElementById(rootId);
            if (!root) {
                Log.debug('local_chatbot app root not found');
                return null;
            }
            if (root.__localChatbotApp) {
                Log.debug('local_chatbot app scaffold already initialized');
                return root.__localChatbotApp;
            }

            const historyAdapter = createHistoryAdapter(bootConfig.userid || 'anon');
            const store = createStore(createInitialState(bootConfig, historyAdapter));
            const refs = getDomRefs();
            const api = ApiClient.create(bootConfig);

            const app = {
                root: root,
                api: api,
                store: store,
                actions: null,
                historyAdapter: historyAdapter,
                config: bootConfig,
                openFilePreview: () => {},
                destroy: null
            };

            const render = () => {
                syncRootDataset(root, store.getState());
                if (store.getState().uiState.ownsMaterialsPreview) {
                    renderMaterialsPreviewDomain(app, refs);
                }
            };

            const unsubscribe = store.subscribe(render);
            app.actions = createActions(app, refs);
            const detachMaterialsPreview = attachMaterialsPreviewDomain(app, refs);

            app.destroy = () => {
                detachMaterialsPreview();
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

            render();
            Log.debug('local_chatbot app scaffold initialized');
            return app;
        }
    };
});
