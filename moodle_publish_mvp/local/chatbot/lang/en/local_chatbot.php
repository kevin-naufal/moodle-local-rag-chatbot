<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'LLM Chat';
$string['settingsheading'] = 'LLM Chat settings';
$string['projectpath'] = 'Project path';
$string['projectpath_desc'] = 'Absolute path to the LLM project folder.';
$string['pythonpath'] = 'Python executable path';
$string['pythonpath_desc'] = 'Absolute path to python executable used by the plugin.';
$string['runnerfile'] = 'Runner file name';
$string['runnerfile_desc'] = 'Backend runner filename inside the project path.';

$string['chatbot:view'] = 'View LLM Chat page.';

$string['materialstitle'] = 'Materials';
$string['uploaddesc'] = 'Select class and topic to load materials from course resources.';
$string['classlabel'] = 'Class';
$string['classplaceholder'] = 'Select class';
$string['topiclabel'] = 'Topic';
$string['topicplaceholder'] = 'Select topic';
$string['uploadedtitle'] = 'Materials';
$string['nofiles'] = 'No materials found for selected class/topic.';
$string['nocoursesavailable'] = 'No enrolled course found';
$string['statusready'] = 'RAG ready';
$string['statusnodocs'] = 'No materials selected';
$string['chatheader'] = 'Chat with class materials';
$string['chatplaceholder'] = 'Ask a question about selected class materials...';
$string['sendbutton'] = 'Send';
$string['chatusagelabel'] = 'Usage';
$string['clearhistorylabel'] = 'Clear history';
$string['clearhistoryconfirm'] = 'Clear this chat history?';
$string['defaultgreeting'] = 'Hello. Select class and topic in the left panel, then ask about the materials.';
$string['thinking'] = 'Thinking...';
$string['chaterror'] = 'Failed to process chat request.';
$string['previewtitle'] = 'Preview';
$string['previewempty'] = 'No preview available.';
$string['previewloading'] = 'Generating preview...';
$string['previewerror'] = 'Failed to generate preview.';
$string['previewopenpdf'] = 'Open PDF in new tab';
$string['previewpdffallback'] = 'If the PDF does not render here, open it in a new tab.';
