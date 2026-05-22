<?php
require_once(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/locallib.php');
require_once($CFG->dirroot . '/enrol/locallib.php');

require_login();

$context = context_system::instance();
require_capability('local/chatbot:view', $context);

$PAGE->set_context($context);
$PAGE->set_url(new moodle_url('/local/chatbot/index.php'));
$PAGE->set_pagelayout('standard');
$PAGE->add_body_class('local-chatbot-fullwidth');
$PAGE->set_title(get_string('pluginname', 'local_chatbot'));
$PAGE->set_heading(get_string('pluginname', 'local_chatbot'));

$chatcourses = [];
$coursetopicsmap = [];
$canmanualupload = local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin();

$classplaceholder = get_string('classplaceholder', 'local_chatbot');
$topicplaceholder = get_string('topicplaceholder', 'local_chatbot');
if (preg_match('/^\[\[[^\]]+\]\]$/', $classplaceholder)) {
    $classplaceholder = 'Select class';
}
if (preg_match('/^\[\[[^\]]+\]\]$/', $topicplaceholder)) {
    $topicplaceholder = 'Select topic';
}

$courses = enrol_get_users_courses(
    (int)$USER->id,
    true,
    'id,shortname,fullname',
    'sortorder ASC'
);

foreach ($courses as $course) {
    if ((int)$course->id === SITEID) {
        continue;
    }
    if (!local_chatbot_user_can_access_course_materials((int)$course->id, (int)$USER->id)) {
        continue;
    }

    $chatcourses[] = $course;
    $topics = local_chatbot_list_course_topics((int)$course->id, (int)$USER->id);
    $courselabel = trim((string)$course->fullname) !== ''
        ? $course->fullname
        : $course->shortname;
    $coursetopicsmap[(string)$course->id] = $topics;
    $coursetopicsmap[$courselabel] = $topics;
}

$PAGE->requires->css('/local/chatbot/styles.css');
$PAGE->requires->js_call_amd('local_chatbot/widget', 'init', [[
    'ajaxurl' => (new moodle_url('/local/chatbot/ajax.php'))->out(false),
    'sesskey' => sesskey(),
    'userid' => (int)$USER->id,
    'chaterror' => get_string('chaterror', 'local_chatbot'),
    'nofiles' => get_string('nofiles', 'local_chatbot'),
    'defaultgreeting' => get_string('defaultgreeting', 'local_chatbot'),
    'thinking' => get_string('thinking', 'local_chatbot'),
    'chatusagelabel' => get_string('chatusagelabel', 'local_chatbot'),
    'previewempty' => get_string('previewempty', 'local_chatbot'),
    'previewloading' => get_string('previewloading', 'local_chatbot'),
    'previewerror' => get_string('previewerror', 'local_chatbot'),
    'previewopenpdf' => get_string('previewopenpdf', 'local_chatbot'),
    'previewpdffallback' => get_string('previewpdffallback', 'local_chatbot'),
    'clearhistoryconfirm' => get_string('clearhistoryconfirm', 'local_chatbot'),
    'statusready' => get_string('statusready', 'local_chatbot'),
    'statusnodocs' => get_string('statusnodocs', 'local_chatbot'),
    'modellabel' => get_string('modellabel', 'local_chatbot'),
    'modeplaceholder' => get_string('modeplaceholder', 'local_chatbot'),
    'mode_llm_only' => get_string('mode_llm_only', 'local_chatbot'),
    'mode_rag_ollama' => get_string('mode_rag_ollama', 'local_chatbot'),
    'mode_rag_bert' => get_string('mode_rag_bert', 'local_chatbot'),
    'mode_rag_msmarco' => get_string('mode_rag_msmarco', 'local_chatbot'),
    'evallabel' => get_string('evallabel', 'local_chatbot'),
    'evalsourcelabel' => get_string('evalsourcelabel', 'local_chatbot'),
    'evalsourcechat' => get_string('evalsourcechat', 'local_chatbot'),
    'evalsourcedataset' => get_string('evalsourcedataset', 'local_chatbot'),
    'evalquestionidlabel' => get_string('evalquestionidlabel', 'local_chatbot'),
    'evalrunidlabel' => get_string('evalrunidlabel', 'local_chatbot'),
    'evaldatasettitle' => get_string('evaldatasettitle', 'local_chatbot'),
    'evaldatasetlabel' => get_string('evaldatasetlabel', 'local_chatbot'),
    'evaldatasetrunslabel' => get_string('evaldatasetrunslabel', 'local_chatbot'),
    'evaldatasetrunbutton' => get_string('evaldatasetrunbutton', 'local_chatbot'),
    'evaldatasetrunning' => get_string('evaldatasetrunning', 'local_chatbot'),
    'evaldatasetsuccess' => get_string('evaldatasetsuccess', 'local_chatbot'),
    'evalformtitle' => get_string('evalformtitle', 'local_chatbot'),
    'evalformscalehelp' => get_string('evalformscalehelp', 'local_chatbot'),
    'evalformcorrectness' => get_string('evalformcorrectness', 'local_chatbot'),
    'evalformgroundedness' => get_string('evalformgroundedness', 'local_chatbot'),
    'evalformrelevance' => get_string('evalformrelevance', 'local_chatbot'),
    'evalforminstructioncompliance' => get_string('evalforminstructioncompliance', 'local_chatbot'),
    'evalformneedalignment' => get_string('evalformneedalignment', 'local_chatbot'),
    'evalformscaffoldingquality' => get_string('evalformscaffoldingquality', 'local_chatbot'),
    'evalformclarity' => get_string('evalformclarity', 'local_chatbot'),
    'evalformcommentlabel' => get_string('evalformcommentlabel', 'local_chatbot'),
    'evalformcommentplaceholder' => get_string('evalformcommentplaceholder', 'local_chatbot'),
    'evalformsubmit' => get_string('evalformsubmit', 'local_chatbot'),
    'evalformsubmitted' => get_string('evalformsubmitted', 'local_chatbot'),
    'evalformscoreplaceholder' => get_string('evalformscoreplaceholder', 'local_chatbot'),
    'evalformsaving' => get_string('evalformsaving', 'local_chatbot'),
    'evalformsaveerror' => get_string('evalformsaveerror', 'local_chatbot'),
    'evalformrequired' => get_string('evalformrequired', 'local_chatbot'),
    'evalformsaveok' => get_string('evalformsaveok', 'local_chatbot'),
    'evaluationmodetitle' => get_string('evaluationmodetitle', 'local_chatbot'),
    'manualuploadrequired' => get_string('manualuploadrequired', 'local_chatbot'),
    'manualuploading' => get_string('manualuploading', 'local_chatbot'),
    'manualuploadsuccess' => get_string('manualuploadsuccess', 'local_chatbot'),
    'manualcleared' => get_string('manualcleared', 'local_chatbot'),
    'manualmodeactive' => get_string('manualmodeactive', 'local_chatbot'),
    'manualuploadreadonly' => get_string('manualuploadreadonly', 'local_chatbot'),
    'canmanualupload' => (bool)$canmanualupload,
    'courseclassplaceholder' => $classplaceholder,
    'coursetopicplaceholder' => $topicplaceholder,
    'coursetopics' => $coursetopicsmap,
]]);

echo $OUTPUT->header();
?>
<div class="local-chatbot-shell">
    <div class="local-chatbot-page">
        <aside class="local-chatbot-sidebar">
            <section class="local-chatbot-card local-chatbot-material-filter">
                <h3><?php echo s(get_string('materialstitle', 'local_chatbot')); ?></h3>
                <p><?php echo s(get_string('uploaddesc', 'local_chatbot')); ?></p>

                <div class="local-chatbot-manual-upload">
                    <h4><?php echo s(get_string('manualuploadtitle', 'local_chatbot')); ?></h4>
                    <label for="local-chatbot-upload-input"><?php echo s(get_string('manualuploadlabel', 'local_chatbot')); ?></label>
                    <input
                        id="local-chatbot-upload-input"
                        class="form-control"
                        type="file"
                        accept=".pdf,.txt,application/pdf,text/plain"
                        multiple
                        <?php echo $canmanualupload ? '' : 'disabled'; ?>
                    />
                    <p class="local-chatbot-upload-help">
                        <?php echo s($canmanualupload
                            ? get_string('manualuploadhelp', 'local_chatbot')
                            : get_string('manualuploadreadonly', 'local_chatbot')); ?>
                    </p>
                    <div class="local-chatbot-upload-actions">
                        <button
                            id="local-chatbot-upload-btn"
                            class="btn btn-primary btn-sm"
                            type="button"
                            <?php echo $canmanualupload ? '' : 'disabled'; ?>
                        >
                            <?php echo s(get_string('manualuploadbutton', 'local_chatbot')); ?>
                        </button>
                        <button
                            id="local-chatbot-clear-upload-btn"
                            class="btn btn-outline-secondary btn-sm"
                            type="button"
                            <?php echo $canmanualupload ? '' : 'disabled'; ?>
                        >
                            <?php echo s(get_string('manualclearbutton', 'local_chatbot')); ?>
                        </button>
                    </div>
                </div>

                <div id="local-chatbot-material-context" class="local-chatbot-material-context local-chatbot-hidden"></div>

                <h4><?php echo s(get_string('topicmaterialstitle', 'local_chatbot')); ?></h4>

                <label for="local-chatbot-chat-class"><?php echo s(get_string('classlabel', 'local_chatbot')); ?></label>
                <select id="local-chatbot-chat-class" class="form-control">
                    <?php if (empty($chatcourses)): ?>
                        <option value=""><?php echo s(get_string('nocoursesavailable', 'local_chatbot')); ?></option>
                    <?php else: ?>
                        <option value=""><?php echo s($classplaceholder); ?></option>
                        <?php foreach ($chatcourses as $course): ?>
                            <?php
                                $courselabel = trim((string)$course->fullname) !== ''
                                    ? $course->fullname
                                    : $course->shortname;
                            ?>
                            <option value="<?php echo (int)$course->id; ?>" data-coursename="<?php echo s($courselabel); ?>">
                                <?php echo s($courselabel); ?>
                            </option>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </select>

                <label for="local-chatbot-chat-topic"><?php echo s(get_string('topiclabel', 'local_chatbot')); ?></label>
                <select id="local-chatbot-chat-topic" class="form-control">
                    <option value=""><?php echo s($topicplaceholder); ?></option>
                </select>
            </section>

            <section class="local-chatbot-card local-chatbot-run-controls">
                <h3><?php echo s(get_string('evaluationmodetitle', 'local_chatbot')); ?></h3>
                <label class="local-chatbot-toggle" for="local-chatbot-eval-mode">
                    <input id="local-chatbot-eval-mode" type="checkbox" />
                    <span><?php echo s(get_string('evallabel', 'local_chatbot')); ?></span>
                </label>

                <div id="local-chatbot-eval-controls" class="local-chatbot-eval-controls local-chatbot-hidden">
                    <div class="local-chatbot-eval-source-group">
                        <label><?php echo s(get_string('evalsourcelabel', 'local_chatbot')); ?></label>
                        <div class="local-chatbot-mode-group">
                            <label class="local-chatbot-mode-option" for="local-chatbot-eval-source-chat">
                                <input
                                    id="local-chatbot-eval-source-chat"
                                    type="radio"
                                    name="local-chatbot-eval-source"
                                    value="chat"
                                    checked
                                />
                                <span><?php echo s(get_string('evalsourcechat', 'local_chatbot')); ?></span>
                            </label>
                            <label class="local-chatbot-mode-option" for="local-chatbot-eval-source-dataset">
                                <input
                                    id="local-chatbot-eval-source-dataset"
                                    type="radio"
                                    name="local-chatbot-eval-source"
                                    value="dataset"
                                />
                                <span><?php echo s(get_string('evalsourcedataset', 'local_chatbot')); ?></span>
                            </label>
                        </div>
                    </div>

                    <div class="local-chatbot-eval-fields">
                        <label><?php echo s(get_string('modellabel', 'local_chatbot')); ?></label>
                        <div class="local-chatbot-mode-group">
                            <label class="local-chatbot-mode-option" for="local-chatbot-mode-llm-only">
                                <input id="local-chatbot-mode-llm-only" type="checkbox" data-mode-value="llm_only" />
                                <span><?php echo s(get_string('mode_llm_only', 'local_chatbot')); ?></span>
                            </label>
                            <label class="local-chatbot-mode-option" for="local-chatbot-mode-rag-ollama">
                                <input id="local-chatbot-mode-rag-ollama" type="checkbox" data-mode-value="rag_ollama" checked />
                                <span><?php echo s(get_string('mode_rag_ollama', 'local_chatbot')); ?></span>
                            </label>
                            <label class="local-chatbot-mode-option" for="local-chatbot-mode-rag-bert">
                                <input id="local-chatbot-mode-rag-bert" type="checkbox" data-mode-value="rag_bert" />
                                <span><?php echo s(get_string('mode_rag_bert', 'local_chatbot')); ?></span>
                            </label>
                            <label class="local-chatbot-mode-option" for="local-chatbot-mode-rag-msmarco">
                                <input id="local-chatbot-mode-rag-msmarco" type="checkbox" data-mode-value="rag_msmarco" />
                                <span><?php echo s(get_string('mode_rag_msmarco', 'local_chatbot')); ?></span>
                            </label>
                        </div>

                        <label for="local-chatbot-question-id"><?php echo s(get_string('evalquestionidlabel', 'local_chatbot')); ?></label>
                        <input id="local-chatbot-question-id" class="form-control" type="text" placeholder="ch03-q01" />

                        <label for="local-chatbot-run-id"><?php echo s(get_string('evalrunidlabel', 'local_chatbot')); ?></label>
                        <input id="local-chatbot-run-id" class="form-control" type="number" min="1" step="1" value="1" />
                    </div>

                    <div class="local-chatbot-eval-dataset">
                        <h4><?php echo s(get_string('evaldatasettitle', 'local_chatbot')); ?></h4>
                        <label for="local-chatbot-eval-dataset-file"><?php echo s(get_string('evaldatasetlabel', 'local_chatbot')); ?></label>
                        <input id="local-chatbot-eval-dataset-file" class="form-control" type="file" accept=".json,application/json" />

                        <label for="local-chatbot-eval-dataset-runs"><?php echo s(get_string('evaldatasetrunslabel', 'local_chatbot')); ?></label>
                        <input id="local-chatbot-eval-dataset-runs" class="form-control" type="number" min="1" step="1" value="1" />

                        <button id="local-chatbot-eval-dataset-run" class="btn btn-primary btn-sm" type="button">
                            <?php echo s(get_string('evaldatasetrunbutton', 'local_chatbot')); ?>
                        </button>
                    </div>
                </div>
            </section>

            <section class="local-chatbot-card local-chatbot-docs">
                <h3><?php echo s(get_string('uploadedtitle', 'local_chatbot')); ?></h3>
                <div id="local-chatbot-files">
                    <p class="local-chatbot-empty"><?php echo s(get_string('nofiles', 'local_chatbot')); ?></p>
                </div>
            </section>
        </aside>

        <section class="local-chatbot-main">
            <header class="local-chatbot-main-header">
                <h3><?php echo s(get_string('chatheader', 'local_chatbot')); ?></h3>
                <div class="local-chatbot-header-meta">
                    <span id="local-chatbot-usage" class="local-chatbot-usage">
                        <?php echo s(get_string('chatusagelabel', 'local_chatbot')); ?>: 0/80
                    </span>
                    <span id="local-chatbot-status" class="local-chatbot-status">
                        <?php echo s(get_string('statusnodocs', 'local_chatbot')); ?>
                    </span>
                    <button id="local-chatbot-clear" class="local-chatbot-clear btn btn-outline-secondary btn-sm" type="button">
                        <?php echo s(get_string('clearhistorylabel', 'local_chatbot')); ?>
                    </button>
                </div>
            </header>

            <div id="local-chatbot-messages" class="local-chatbot-messages"></div>

            <div class="local-chatbot-composer">
                <input
                    id="local-chatbot-input"
                    type="text"
                    placeholder="<?php echo s(get_string('chatplaceholder', 'local_chatbot')); ?>"
                />
                <button id="local-chatbot-send" class="btn btn-secondary" type="button">
                    <?php echo s(get_string('sendbutton', 'local_chatbot')); ?>
                </button>
            </div>
        </section>

        <section class="local-chatbot-preview">
            <header class="local-chatbot-preview-header">
                <h3><?php echo s(get_string('previewtitle', 'local_chatbot')); ?></h3>
                <span id="local-chatbot-preview-name" class="local-chatbot-preview-name">-</span>
            </header>
            <div id="local-chatbot-preview-body" class="local-chatbot-preview-body">
                <p class="local-chatbot-empty"><?php echo s(get_string('previewempty', 'local_chatbot')); ?></p>
            </div>
        </section>
    </div>
</div>
<?php
echo $OUTPUT->footer();

