define([], function() {
    const parseJsonResponse = async(response) => {
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

    const appendFormValue = (form, key, value) => {
        if (value === undefined || value === null) {
            return;
        }
        form.append(key, value);
    };

    const buildFormData = (action, payload = {}) => {
        const form = new FormData();
        form.append('action', action);

        Object.keys(payload).forEach((key) => {
            const value = payload[key];
            if (Array.isArray(value)) {
                value.forEach((item) => appendFormValue(form, key, item));
                return;
            }
            appendFormValue(form, key, value);
        });

        return form;
    };

    const create = (config = {}) => {
        const ajaxUrl = String(config.ajaxurl || '').trim();
        const sesskey = String(config.sesskey || '').trim();

        if (!ajaxUrl || !sesskey) {
            throw new Error('Chatbot API client requires ajaxurl and sesskey.');
        }

        const post = async(action, payload = {}) => {
            const form = buildFormData(action, Object.assign({sesskey: sesskey}, payload));
            const response = await fetch(ajaxUrl, {
                method: 'POST',
                body: form,
                credentials: 'same-origin',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });
            return parseJsonResponse(response);
        };

        return {
            post: post,
            listFiles: () => post('list_files'),
            setMaterialContext: (courseid, topic) => post('set_material_context', {
                courseid: String(courseid || ''),
                topic: String(topic || '')
            }),
            uploadMaterials: (files = []) => {
                const form = new FormData();
                form.append('action', 'upload');
                form.append('sesskey', sesskey);
                Array.from(files).forEach((file) => {
                    form.append('documents[]', file);
                });
                return fetch(ajaxUrl, {
                    method: 'POST',
                    body: form,
                    credentials: 'same-origin',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                }).then(parseJsonResponse);
            },
            clearUploadedMaterials: () => post('clear_uploaded_materials'),
            fileContent: (filename) => post('file_content', {
                filename: String(filename || '')
            }),
            refreshSelectedEmbedding: (filename) => post('refresh_selected_embedding', {
                filename: String(filename || '')
            }),
            chat: (payload = {}) => post('chat', payload)
        };
    };

    return {
        create: create
    };
});
