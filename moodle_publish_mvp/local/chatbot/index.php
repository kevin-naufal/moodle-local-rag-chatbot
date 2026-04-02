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

$isteacher = local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin();
$showpractice = !$isteacher || is_siteadmin((int)$USER->id);
$chatcourses = [];
$teachercourses = [];
$coursetopicsmap = [];
$coursepdfsmap = [];

$assignmentclassplaceholder = get_string('assignmentclassplaceholder', 'local_chatbot');
$assignmenttopicplaceholder = get_string('assignmenttopicplaceholder', 'local_chatbot');
$assignmenttopicloading = get_string('assignmenttopicloading', 'local_chatbot');
$assignmenttopicempty = get_string('assignmenttopicempty', 'local_chatbot');
$assignmentpdflabel = get_string('assignmentpdf', 'local_chatbot');
$assignmentpdfplaceholder = get_string('assignmentpdfplaceholder', 'local_chatbot');
$assignmentpdfloading = get_string('assignmentpdfloading', 'local_chatbot');
$assignmentpdfempty = get_string('assignmentpdfempty', 'local_chatbot');
if (preg_match('/^\[\[[^\]]+\]\]$/', $assignmentclassplaceholder)) {
    $assignmentclassplaceholder = 'Pilih kelas';
}
if (preg_match('/^\[\[[^\]]+\]\]$/', $assignmenttopicplaceholder)) {
    $assignmenttopicplaceholder = 'Pilih topik';
}
if (preg_match('/^\[\[[^\]]+\]\]$/', $assignmenttopicloading)) {
    $assignmenttopicloading = 'Loading topics...';
}
if (preg_match('/^\[\[[^\]]+\]\]$/', $assignmenttopicempty)) {
    $assignmenttopicempty = 'No topics found in this class';
}
if (preg_match('/^\[\[[^\]]+\]\]$/', $assignmentpdflabel)) {
    $assignmentpdflabel = 'Materi';
}
if (preg_match('/^\[\[[^\]]+\]\]$/', $assignmentpdfplaceholder)) {
    $assignmentpdfplaceholder = 'Pilih PDF';
}
if (preg_match('/^\[\[[^\]]+\]\]$/', $assignmentpdfloading)) {
    $assignmentpdfloading = 'Loading PDFs...';
}
if (preg_match('/^\[\[[^\]]+\]\]$/', $assignmentpdfempty)) {
    $assignmentpdfempty = 'No PDF resource found in this class';
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
    $coursecontext = context_course::instance((int)$course->id);
    $canaccesscourse = local_chatbot_user_can_access_course_materials((int)$course->id, (int)$USER->id);
    $canmanagecourse = has_capability('moodle/course:update', $coursecontext, (int)$USER->id);
    if (!$canaccesscourse) {
        continue;
    }

    $chatcourses[] = $course;
    if ($canmanagecourse || is_siteadmin((int)$USER->id)) {
        $teachercourses[] = $course;
    }

    $topics = local_chatbot_list_course_topics((int)$course->id, (int)$USER->id);
    $pdfs = local_chatbot_list_course_pdfs((int)$course->id, (int)$USER->id);
    $coursetopicsmap[(string)$course->id] = $topics;
    $courselabel = trim((string)$course->fullname) !== ''
        ? $course->fullname
        : $course->shortname;
    $coursetopicsmap[$courselabel] = $topics;
    $coursepdfsmap[(string)$course->id] = $pdfs;
    $coursepdfsmap[$courselabel] = $pdfs;
}

$PAGE->requires->css('/local/chatbot/styles.css');
$PAGE->requires->js_call_amd('local_chatbot/widget', 'init', [[
    'ajaxurl' => (new moodle_url('/local/chatbot/ajax.php'))->out(false),
    'savedrafturl' => (new moodle_url('/local/chatbot/save_draft.php'))->out(false),
    'publishurl' => (new moodle_url('/local/chatbot/publish.php'))->out(false),
    'sesskey' => sesskey(),
    'viewurlbase' => (new moodle_url('/local/chatbot/view.php'))->out(false),
    'chaterror' => get_string('chaterror', 'local_chatbot'),
    'nofiles' => get_string('nofiles', 'local_chatbot'),
    'defaultgreeting' => get_string('defaultgreeting', 'local_chatbot'),
    'thinking' => get_string('thinking', 'local_chatbot'),
    'uploading' => get_string('uploading', 'local_chatbot'),
    'uploadfailed' => get_string('uploadfailed', 'local_chatbot'),
    'chatusagelabel' => get_string('chatusagelabel', 'local_chatbot'),
    'previewtitle' => get_string('previewtitle', 'local_chatbot'),
    'previewempty' => get_string('previewempty', 'local_chatbot'),
    'previewloading' => get_string('previewloading', 'local_chatbot'),
    'previewerror' => get_string('previewerror', 'local_chatbot'),
    'clearhistorylabel' => get_string('clearhistorylabel', 'local_chatbot'),
    'clearhistoryconfirm' => get_string('clearhistoryconfirm', 'local_chatbot'),
    'tabchat' => get_string('tabchat', 'local_chatbot'),
    'tabassignment' => get_string('tabassignment', 'local_chatbot'),
    'tabpractice' => get_string('tabpractice', 'local_chatbot'),
    'assignmentgenerate' => get_string('assignmentgenerate', 'local_chatbot'),
    'assignmentregenerate' => get_string('assignmentregenerate', 'local_chatbot'),
    'assignmentpublish' => get_string('assignmentpublish', 'local_chatbot'),
    'assignmentpublished' => get_string('assignmentpublished', 'local_chatbot'),
    'assignmentpublishing' => get_string('assignmentpublishing', 'local_chatbot'),
    'assignmentpublisherror' => get_string('assignmentpublisherror', 'local_chatbot'),
    'assignmentgeneratedfirst' => get_string('assignmentgeneratedfirst', 'local_chatbot'),
    'assignmentselectclassfirst' => get_string('assignmentselectclassfirst', 'local_chatbot'),
    'assignmentplaceholder' => get_string('assignmentplaceholder', 'local_chatbot'),
    'assignmenttopicplaceholder' => $assignmenttopicplaceholder,
    'assignmenttopicloading' => $assignmenttopicloading,
    'assignmenttopicempty' => $assignmenttopicempty,
    'assignmentpdfplaceholder' => $assignmentpdfplaceholder,
    'assignmentpdfloading' => $assignmentpdfloading,
    'assignmentpdfempty' => $assignmentpdfempty,
    'practicegenerate' => get_string('practicegenerate', 'local_chatbot'),
    'practiceplaceholder' => get_string('practiceplaceholder', 'local_chatbot'),
    'practicepublish' => get_string('practicepublish', 'local_chatbot'),
    'practicepublished' => get_string('practicepublished', 'local_chatbot'),
    'practicepublishing' => get_string('practicepublishing', 'local_chatbot'),
    'practicepublisherror' => get_string('practicepublisherror', 'local_chatbot'),
    'practicegeneratedfirst' => get_string('practicegeneratedfirst', 'local_chatbot'),
    'roleteacheronly' => get_string('roleteacheronly', 'local_chatbot'),
    'coursetopics' => $coursetopicsmap,
    'coursepdfs' => $coursepdfsmap,
    'isteacher' => $isteacher ? 1 : 0,
    'userid' => (int)$USER->id,
]]);

$files = [];

echo $OUTPUT->header();
?>
<div class="local-chatbot-shell">
    <nav class="local-chatbot-tabs" aria-label="LLM Tutor Navigation">
        <button class="local-chatbot-tab active" type="button" data-tab="chat">
            <?php echo s(get_string('tabchat', 'local_chatbot')); ?>
        </button>
        <?php if ($isteacher): ?>
            <button class="local-chatbot-tab" type="button" data-tab="assignment">
                <?php echo s(get_string('tabassignment', 'local_chatbot')); ?>
            </button>
        <?php endif; ?>
        <?php if ($showpractice): ?>
            <button class="local-chatbot-tab" type="button" data-tab="practice">
                <?php echo s(get_string('tabpractice', 'local_chatbot')); ?>
            </button>
        <?php endif; ?>
    </nav>
    <section id="local-chatbot-panel-chat" class="local-chatbot-tab-panel active">
        <div class="local-chatbot-page">
            <aside class="local-chatbot-sidebar">
                <section class="local-chatbot-card local-chatbot-material-filter">
                    <h3><?php echo s(get_string('assignmentpdf', 'local_chatbot')); ?></h3>
                    <p><?php echo s(get_string('uploaddesc', 'local_chatbot')); ?></p>
                    <label for="local-chatbot-chat-class"><?php echo s(get_string('assignmentclass', 'local_chatbot')); ?></label>
                    <select id="local-chatbot-chat-class" class="form-control">
                        <?php if (empty($chatcourses)): ?>
                            <option value=""><?php echo s(get_string('nocoursesavailable', 'local_chatbot')); ?></option>
                        <?php else: ?>
                            <option value=""><?php echo s($assignmentclassplaceholder); ?></option>
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

                    <label for="local-chatbot-chat-topic"><?php echo s(get_string('assignmenttopic', 'local_chatbot')); ?></label>
                    <select id="local-chatbot-chat-topic" class="form-control">
                        <option value=""><?php echo s($assignmenttopicplaceholder); ?></option>
                    </select>
                </section>

                <section class="local-chatbot-card local-chatbot-docs">
                    <h3><?php echo s(get_string('uploadedtitle', 'local_chatbot')); ?></h3>
                    <div id="local-chatbot-files">
                        <?php if (empty($files)): ?>
                            <p class="local-chatbot-empty"><?php echo s(get_string('nofiles', 'local_chatbot')); ?></p>
                        <?php else: ?>
                            <?php foreach ($files as $file): ?>
                                <button
                                    class="local-chatbot-file-item"
                                    type="button"
                                    data-file="<?php echo s($file['name']); ?>"
                                >
                                    <span><?php echo s($file['name']); ?></span>
                                </button>
                            <?php endforeach; ?>
                        <?php endif; ?>
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
                            <?php echo empty($files) ? s(get_string('statusnodocs', 'local_chatbot')) : s(get_string('statusready', 'local_chatbot')); ?>
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
                    <button id="local-chatbot-send" class="btn btn-secondary" type="button">Send</button>
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
    </section>

    <?php if ($isteacher): ?>
        <section id="local-chatbot-panel-assignment" class="local-chatbot-tab-panel">
            <div class="local-chatbot-tool-grid">
                <section class="local-chatbot-card local-chatbot-tool-form">
                    <h3><?php echo s(get_string('assignmenttitle', 'local_chatbot')); ?></h3>
                    <p><?php echo s(get_string('assignmentdesc', 'local_chatbot')); ?></p>

                    <label for="local-chatbot-assign-class"><?php echo s(get_string('assignmentclass', 'local_chatbot')); ?></label>
                    <select id="local-chatbot-assign-class" class="form-control">
                        <?php if (empty($teachercourses)): ?>
                            <option value=""><?php echo s(get_string('assignmentnocourses', 'local_chatbot')); ?></option>
                        <?php else: ?>
                            <option value=""><?php echo s($assignmentclassplaceholder); ?></option>
                            <?php foreach ($teachercourses as $course): ?>
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

                    <label for="local-chatbot-assign-topic"><?php echo s(get_string('assignmenttopic', 'local_chatbot')); ?></label>
                    <select id="local-chatbot-assign-topic" class="form-control">
                        <option value=""><?php echo s($assignmenttopicplaceholder); ?></option>
                    </select>

                    <label for="local-chatbot-assign-pdf"><?php echo s($assignmentpdflabel); ?></label>
                    <select id="local-chatbot-assign-pdf" class="form-control">
                        <option value=""><?php echo s($assignmentpdfplaceholder); ?></option>
                    </select>

                    <label for="local-chatbot-assign-type"><?php echo s(get_string('assignmenttype', 'local_chatbot')); ?></label>
                    <select id="local-chatbot-assign-type" class="form-control">
                        <option value="essay">Essay</option>
                        <option value="multiple-choice">Multiple Choice</option>
                    </select>
                    <label class="local-chatbot-checkbox-row">
                        <input id="local-chatbot-assign-essay-autograde" type="checkbox" checked="checked" />
                        <span><?php echo s(get_string('assignmentessayautograde', 'local_chatbot')); ?></span>
                    </label>

                    <label for="local-chatbot-assign-count"><?php echo s(get_string('assignmentcount', 'local_chatbot')); ?></label>
                    <input id="local-chatbot-assign-count" type="number" min="1" max="10" class="form-control" value="5" />

                    <label for="local-chatbot-assign-notes"><?php echo s(get_string('assignmentnotes', 'local_chatbot')); ?></label>
                    <textarea id="local-chatbot-assign-notes" class="form-control" rows="4" placeholder="Tambahkan batasan, format output, atau rubrik khusus"></textarea>

                    <div class="local-chatbot-tool-actions">
                        <button id="local-chatbot-assign-generate" class="btn btn-primary" type="button">
                            <?php echo s(get_string('assignmentgenerate', 'local_chatbot')); ?>
                        </button>
                        <button id="local-chatbot-assign-regenerate" class="btn btn-outline-secondary" type="button">
                            <?php echo s(get_string('assignmentregenerate', 'local_chatbot')); ?>
                        </button>
                        <button id="local-chatbot-assign-publish" class="btn btn-success" type="button">
                            <?php echo s(get_string('assignmentpublish', 'local_chatbot')); ?>
                        </button>
                    </div>
                </section>

                <section class="local-chatbot-card local-chatbot-tool-preview">
                    <h3><?php echo s(get_string('assignmentpreview', 'local_chatbot')); ?></h3>
                    <div id="local-chatbot-assign-preview" class="local-chatbot-generated-text">
                        <?php echo s(get_string('assignmentplaceholder', 'local_chatbot')); ?>
                    </div>
                </section>
            </div>
        </section>
    <?php endif; ?>

    <?php if ($showpractice): ?>
        <section id="local-chatbot-panel-practice" class="local-chatbot-tab-panel">
            <div class="local-chatbot-tool-grid">
                <section class="local-chatbot-card local-chatbot-tool-form">
                    <h3><?php echo s(get_string('practicetitle', 'local_chatbot')); ?></h3>
                    <p><?php echo s(get_string('practicedesc', 'local_chatbot')); ?></p>

                    <label for="local-chatbot-practice-class"><?php echo s(get_string('assignmentclass', 'local_chatbot')); ?></label>
                    <select id="local-chatbot-practice-class" class="form-control">
                        <?php if (empty($chatcourses)): ?>
                            <option value=""><?php echo s(get_string('nocoursesavailable', 'local_chatbot')); ?></option>
                        <?php else: ?>
                            <option value=""><?php echo s($assignmentclassplaceholder); ?></option>
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

                    <label for="local-chatbot-practice-topic"><?php echo s(get_string('assignmenttopic', 'local_chatbot')); ?></label>
                    <select id="local-chatbot-practice-topic" class="form-control">
                        <option value=""><?php echo s($assignmenttopicplaceholder); ?></option>
                    </select>

                    <label for="local-chatbot-practice-pdf"><?php echo s($assignmentpdflabel); ?></label>
                    <select id="local-chatbot-practice-pdf" class="form-control">
                        <option value=""><?php echo s($assignmentpdfplaceholder); ?></option>
                    </select>

                    <label for="local-chatbot-practice-count"><?php echo s(get_string('assignmentcount', 'local_chatbot')); ?></label>
                    <input id="local-chatbot-practice-count" type="number" min="1" max="10" class="form-control" value="5" />

                    <div class="local-chatbot-tool-actions">
                        <button id="local-chatbot-practice-generate" class="btn btn-primary" type="button">
                            <?php echo s(get_string('practicegenerate', 'local_chatbot')); ?>
                        </button>
                        <button id="local-chatbot-practice-publish" class="btn btn-success" type="button">
                            <?php echo s(get_string('practicepublish', 'local_chatbot')); ?>
                        </button>
                    </div>
                </section>

                <section class="local-chatbot-card local-chatbot-tool-preview">
                    <h3><?php echo s(get_string('practicepreview', 'local_chatbot')); ?></h3>
                    <div id="local-chatbot-practice-preview" class="local-chatbot-generated-text">
                        <?php echo s(get_string('practiceplaceholder', 'local_chatbot')); ?>
                    </div>
                </section>
            </div>
        </section>
    <?php endif; ?>

</div>
<?php
echo $OUTPUT->footer();

