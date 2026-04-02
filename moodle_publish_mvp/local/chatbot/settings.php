<?php
defined('MOODLE_INTERNAL') || die();

if ($hassiteconfig) {
    $settings = new admin_settingpage('local_chatbot_settings', get_string('pluginname', 'local_chatbot'));

    $settings->add(new admin_setting_heading(
        'local_chatbot/settingsheading',
        get_string('settingsheading', 'local_chatbot'),
        ''
    ));

    $settings->add(new admin_setting_configtext(
        'local_chatbot/projectpath',
        get_string('projectpath', 'local_chatbot'),
        get_string('projectpath_desc', 'local_chatbot'),
        'C:\\Users\\Kevin\\Downloads\\my-llm',
        PARAM_RAW_TRIMMED
    ));

    $settings->add(new admin_setting_configtext(
        'local_chatbot/pythonpath',
        get_string('pythonpath', 'local_chatbot'),
        get_string('pythonpath_desc', 'local_chatbot'),
        'C:\\Users\\Kevin\\Downloads\\my-llm\\.venv\\Scripts\\python.exe',
        PARAM_RAW_TRIMMED
    ));

    $settings->add(new admin_setting_configtext(
        'local_chatbot/runnerfile',
        get_string('runnerfile', 'local_chatbot'),
        get_string('runnerfile_desc', 'local_chatbot'),
        'app/moodle_rag_runner.py',
        PARAM_RAW_TRIMMED
    ));

    $ADMIN->add('localplugins', $settings);
}
