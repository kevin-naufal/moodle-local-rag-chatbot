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
$string['uploaddesc'] = 'Select class/topic or upload manual PDF/TXT materials to define the active RAG corpus.';
$string['manualuploadtitle'] = 'Manual upload';
$string['manualuploadlabel'] = 'Upload PDF/TXT materials';
$string['manualuploadbutton'] = 'Upload materials';
$string['manualclearbutton'] = 'Clear uploaded materials';
$string['manualuploadhelp'] = 'Manual upload becomes the active corpus and temporarily disables topic selection.';
$string['manualuploadreadonly'] = 'Upload is visible here, but only teacher-like users can change the active corpus.';
$string['manualuploadrequired'] = 'Choose at least one PDF or TXT file first.';
$string['manualuploading'] = 'Uploading materials...';
$string['manualuploadsuccess'] = 'Manual materials uploaded. Topic selection is now disabled.';
$string['manualcleared'] = 'Manual uploaded materials cleared. Topic selection is enabled again.';
$string['manualmodeactive'] = 'Manual upload is active. Clear uploaded materials to use class/topic materials again.';
$string['topicmaterialstitle'] = 'Class/topic materials';
$string['classlabel'] = 'Class';
$string['classplaceholder'] = 'Select class';
$string['topiclabel'] = 'Topic';
$string['topicplaceholder'] = 'Select topic';
$string['executiontitle'] = 'Execution';
$string['evaluationmodetitle'] = 'Evaluation mode';
$string['modellabel'] = 'Mode';
$string['modeplaceholder'] = 'Select mode';
$string['mode_llm_only'] = 'LLM only';
$string['mode_rag_ollama'] = 'RAG Ollama';
$string['mode_rag_bert'] = 'RAG BERT';
$string['evallabel'] = 'Enable evaluation mode';
$string['evalquestionidlabel'] = 'Question ID';
$string['evalrunidlabel'] = 'Run ID';
$string['evaldatasettitle'] = 'Answer-run dataset';
$string['evaldatasetlabel'] = 'Upload dataset JSON';
$string['evaldatasetrunslabel'] = 'Runs per question';
$string['evaldatasetrunbutton'] = 'Run answer-run dataset';
$string['evaldatasetrunning'] = 'Running answer-run dataset...';
$string['evaldatasetsuccess'] = 'Answer-run dataset finished.';
$string['uploadedtitle'] = 'Materials';
$string['nofiles'] = 'No active materials loaded.';
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
