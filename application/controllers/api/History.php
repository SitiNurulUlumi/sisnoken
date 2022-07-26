<?php
if (!defined('BASEPATH')) exit('No direct script access allowed');

// Load the Rest Controller library
require APPPATH . '/libraries/REST_Controller.php';

class History extends REST_Controller
{

    public function __construct()
    {
        parent::__construct();

        // Load the user model
        $this->load->model('history_model');
    }

    public function history_get()
    {
        // Returns all the users data if the id not specified,
        // Otherwise, a single user will be returned.
        $history = $this->history_model->getRows();

        // Check if the user data exists
        if (!empty($history)) {
            // Set the response and exit
            //OK (200) being the HTTP response code
            $this->response($history, REST_Controller::HTTP_OK);
        } else {
            // Set the response and exit
            //NOT_FOUND (404) being the HTTP response code
            $this->response([
                'status' => FALSE,
                'message' => 'No Data.'
            ], REST_Controller::HTTP_NOT_FOUND);
        }
    }
}
