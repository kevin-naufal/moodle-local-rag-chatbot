define(['core/log'], function(Log) {
    const MAX_HISTORY = 80;
    const MAX_USER_MESSAGES = 80;

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

    const userMessageCount = (history) => history.filter((entry) => entry.type === 'user').length;

            const trimHistory = (history) => {
                const next = Array.isArray(history) ? history.slice() : [];
                while (userMessageCount(next) > MAX_USER_MESSAGES && next.length > 0) {
                    next.shift();
                }
                return next.slice(-MAX_HISTORY);
            };

            const buildConversationContext = (items, limit = 6) => {
                const list = Array.isArray(items) ? items.slice(-limit) : [];
                return list
                    .map((entry) => {
                        const type = String((entry && entry.type) || '').trim().toLowerCase();
                        const text = String((entry && entry.text) || '').trim();
                        if (!text) {
                            return null;
                        }
                        if (type === 'user') {
                            return {role: 'user', text: text};
                        }
                        if (type === 'assistant' || type === 'bot') {
                            return {role: 'assistant', text: text};
                        }
                        return null;
                    })
                    .filter(Boolean);
            };

    const postForm = async (url, data) => {
        const response = await fetch(url, {
            method: 'POST',
            body: data,
            credentials: 'same-origin',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
        const raw = await response.text();
        if (!raw || !raw.trim()) {
            throw new Error(`Empty response from server (HTTP ${response.status}).`);
        }
        try {
            return JSON.parse(raw);
        } catch (err) {
            const snippet = raw.slice(0, 180).replace(/\s+/g, ' ').trim();
            throw new Error(`Invalid JSON response (HTTP ${response.status}): ${snippet || 'no payload'}`);
        }
    };

    const appendMessageDom = (text, type, sources) => {
        const messages = document.getElementById('local-chatbot-messages');
        if (!messages) {
            return null;
        }

        const item = document.createElement('div');
        item.className = `local-chatbot-message ${type}`;
        item.textContent = String(text || '');
        messages.appendChild(item);

        if (Array.isArray(sources) && sources.length > 0) {
            const source = document.createElement('div');
            source.className = 'local-chatbot-source';
            source.textContent = `source: ${sources.join(', ')}`;
            item.appendChild(source);
        }

        messages.scrollTop = messages.scrollHeight;
        return item;
    };

    const renderFiles = (files, nofiles, onFileClick, selectedFile) => {
        const wrap = document.getElementById('local-chatbot-files');
        if (!wrap) {
            return;
        }

        if (!Array.isArray(files) || files.length === 0) {
            wrap.innerHTML = `<p class="local-chatbot-empty">${nofiles}</p>`;
            return;
        }

        wrap.innerHTML = '';
        files.forEach((file) => {
            const name = String((file && file.name) || '');
            if (name === '') {
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
            wrap.appendChild(item);
        });
    };

    return {
        init: function(config) {
            Log.debug('local_chatbot page initialized');

            const input = document.getElementById('local-chatbot-input');
            const sendBtn = document.getElementById('local-chatbot-send');
            const clearBtn = document.getElementById('local-chatbot-clear');
            const usageWrap = document.getElementById('local-chatbot-usage');
            const statusWrap = document.getElementById('local-chatbot-status');
            const previewBody = document.getElementById('local-chatbot-preview-body');
            const previewName = document.getElementById('local-chatbot-preview-name');
            const chatClassInput = document.getElementById('local-chatbot-chat-class');
            const chatTopicInput = document.getElementById('local-chatbot-chat-topic');

            const storageKey = `local_chatbot_history_u${config.userid || 'anon'}`;
            const configuredCourseTopics = (config && typeof config.coursetopics === 'object' && config.coursetopics !== null)
                ? config.coursetopics
                : {};

            let history = trimHistory(safeReadHistory(storageKey));
            let selectedFile = null;
            let activeFiles = [];

            const updateUsage = () => {
                if (!usageWrap) {
                    return;
                }
                usageWrap.textContent = `${config.chatusagelabel || 'Usage'}: ${userMessageCount(history)}/${MAX_USER_MESSAGES}`;
            };

            const persistHistory = () => {
                history = trimHistory(history);
                safeWriteHistory(storageKey, history);
                updateUsage();
            };

            const renderHistory = () => {
                const messages = document.getElementById('local-chatbot-messages');
                if (!messages) {
                    return;
                }
                messages.innerHTML = '';
                if (!history.length) {
                    appendMessageDom(config.defaultgreeting || '', 'assistant', []);
                    return;
                }
                history.forEach((entry) => {
                    appendMessageDom(entry.text || '', entry.type || 'assistant', entry.sources || []);
                });
            };

            const resetPreview = (message) => {
                if (previewName) {
                    previewName.textContent = '-';
                }
                if (previewBody) {
                    previewBody.innerHTML = `<p class="local-chatbot-empty">${message}</p>`;
                }
            };

            const setStatus = (text) => {
                if (statusWrap) {
                    statusWrap.textContent = String(text || '');
                }
            };

            const populateTopicOptions = (courseValue) => {
                if (!chatTopicInput) {
                    return;
                }
                const topics = configuredCourseTopics[String(courseValue || '')] || [];
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
            };

            const loadMaterialContext = async () => {
                const courseid = chatClassInput ? String(chatClassInput.value || '').trim() : '';
                const topic = chatTopicInput ? String(chatTopicInput.value || '').trim() : '';

                selectedFile = null;
                activeFiles = [];
                renderFiles([], config.nofiles || 'No materials found.', () => {}, selectedFile);
                resetPreview(config.previewempty || 'No preview available.');

                if (!courseid || !topic) {
                    setStatus(config.statusnodocs || 'No materials selected');
                    return;
                }

                setStatus(config.previewloading || 'Loading...');
                const form = new FormData();
                form.append('action', 'set_material_context');
                form.append('sesskey', config.sesskey);
                form.append('courseid', courseid);
                form.append('topic', topic);

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.chaterror || 'Failed to load materials.'));
                    }
                    activeFiles = Array.isArray(payload.files) ? payload.files : [];
                    renderFiles(activeFiles, config.nofiles || 'No materials found.', openFilePreview, selectedFile);
                    setStatus(activeFiles.length > 0 ? (config.statusready || 'RAG ready') : (config.statusnodocs || 'No materials selected'));
                } catch (err) {
                    setStatus(config.chaterror || 'Failed to process chat request.');
                    renderFiles([], config.nofiles || 'No materials found.', () => {}, selectedFile);
                    resetPreview(config.previewerror || 'Failed to generate preview.');
                }
            };

            const openFilePreview = async (filename) => {
                selectedFile = filename;
                renderFiles(activeFiles, config.nofiles || 'No materials found.', openFilePreview, selectedFile);
                if (previewName) {
                    previewName.textContent = filename;
                }
                if (previewBody) {
                    previewBody.innerHTML = `<p class="local-chatbot-empty">${config.previewloading || 'Loading preview...'}</p>`;
                }

                const form = new FormData();
                form.append('action', 'file_content');
                form.append('sesskey', config.sesskey);
                form.append('filename', filename);

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.previewerror || 'Failed to generate preview.'));
                    }

                    if (!previewBody) {
                        return;
                    }

                    if (payload.filetype === 'pdf' && payload.viewurl) {
                        previewBody.innerHTML = '';
                        const frame = document.createElement('iframe');
                        frame.src = payload.viewurl;
                        frame.title = filename;
                        frame.loading = 'lazy';
                        previewBody.appendChild(frame);

                        const note = document.createElement('p');
                        note.className = 'local-chatbot-preview-note';
                        note.textContent = String(
                            config.previewpdffallback || 'If the PDF does not render here, open it in a new tab.'
                        );
                        previewBody.appendChild(note);

                        const link = document.createElement('a');
                        link.href = payload.viewurl;
                        link.target = '_blank';
                        link.rel = 'noopener noreferrer';
                        link.textContent = String(config.previewopenpdf || 'Open PDF in new tab');
                        previewBody.appendChild(link);
                        return;
                    }

                    const pre = document.createElement('pre');
                    pre.textContent = String(payload.content || '');
                    previewBody.innerHTML = '';
                    previewBody.appendChild(pre);
                } catch (err) {
                    resetPreview(config.previewerror || 'Failed to generate preview.');
                }
            };

            const sendMessage = async () => {
                const question = input ? String(input.value || '').trim() : '';
                if (!question) {
                    return;
                }
                const conversationContext = buildConversationContext(history);

                history.push({type: 'user', text: question, sources: []});
                persistHistory();
                renderHistory();
                input.value = '';

                const placeholder = appendMessageDom(config.thinking || 'Thinking...', 'assistant', []);
                if (sendBtn) {
                    sendBtn.disabled = true;
                }
                if (input) {
                    input.disabled = true;
                }

                const form = new FormData();
                form.append('action', 'chat');
                form.append('sesskey', config.sesskey);
                form.append('question', question);
                form.append('history', JSON.stringify(conversationContext));
                if (chatClassInput && String(chatClassInput.value || '').trim() !== '') {
                    form.append('courseid', String(chatClassInput.value || '').trim());
                }
                if (chatTopicInput && String(chatTopicInput.value || '').trim() !== '') {
                    form.append('topic', String(chatTopicInput.value || '').trim());
                }

                try {
                    const payload = await postForm(config.ajaxurl, form);
                    if (!payload || !payload.ok) {
                        throw new Error((payload && payload.error) || (config.chaterror || 'Failed to process chat request.'));
                    }

                    if (placeholder && placeholder.parentNode) {
                        placeholder.parentNode.removeChild(placeholder);
                    }

                    const answer = String(payload.answer || '');
                    const sources = Array.isArray(payload.sources) ? payload.sources : [];
                    history.push({type: 'assistant', text: answer, sources: sources});
                    persistHistory();
                    renderHistory();
                } catch (err) {
                    if (placeholder) {
                        placeholder.textContent = err && err.message ? err.message : (config.chaterror || 'Failed to process chat request.');
                    } else {
                        history.push({
                            type: 'assistant',
                            text: err && err.message ? err.message : (config.chaterror || 'Failed to process chat request.'),
                            sources: []
                        });
                        persistHistory();
                        renderHistory();
                    }
                } finally {
                    if (sendBtn) {
                        sendBtn.disabled = false;
                    }
                    if (input) {
                        input.disabled = false;
                        input.focus();
                    }
                }
            };

            if (chatClassInput) {
                chatClassInput.addEventListener('change', () => {
                    populateTopicOptions(String(chatClassInput.value || '').trim());
                    if (chatTopicInput) {
                        chatTopicInput.value = '';
                    }
                    loadMaterialContext();
                });
            }

            if (chatTopicInput) {
                chatTopicInput.addEventListener('change', () => {
                    loadMaterialContext();
                });
            }

            if (sendBtn) {
                sendBtn.addEventListener('click', sendMessage);
            }

            if (input) {
                input.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        sendMessage();
                    }
                });
            }

            if (clearBtn) {
                clearBtn.addEventListener('click', () => {
                    if (!window.confirm(config.clearhistoryconfirm || 'Clear this chat history?')) {
                        return;
                    }
                    history = [];
                    persistHistory();
                    renderHistory();
                });
            }

            populateTopicOptions(chatClassInput ? String(chatClassInput.value || '').trim() : '');
            renderFiles([], config.nofiles || 'No materials found.', () => {}, selectedFile);
            resetPreview(config.previewempty || 'No preview available.');
            setStatus(config.statusnodocs || 'No materials selected');
            persistHistory();
            renderHistory();
        }
    };
});
