<?php

class Profile extends CI_Controller
{

    public function __construct()
    {
        parent::__construct();

        // load model
        $this->load->model('user_model');

        // load helper
        $this->load->helper(array('form', 'url', 'timezone_helper'));

        // load library
        $this->load->library(array('session', 'form_validation'));

        // autentifikasi
        if (!$this->session->userdata('is_logged_in')) {
            redirect("home");
        }
    }

    public function index()
    {
        $data['title'] = 'Profil';
        $data['subtitle'] = 'Profil akun anda';
        $data['profile'] = $this->user_model->get_by_id($this->session->userdata('user_id'));

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('profile/index', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('profile/script', $data);
    }

    public function ajax_save()
    {
        $this->_validate();
        $data = array(
            'password' => md5($this->input->post('repassword')),
        );
        $this->user_model->update(array('id' => $this->session->userdata('user_id')), $data);
        echo json_encode(array("status" => TRUE));
    }



    private function _validate()
    {
        $data = array();
        $data['error_string'] = array();
        $data['inputerror'] = array();
        $data['status'] = TRUE;

        if ($this->input->post('password') == '') {
            $data['inputerror'][] = 'password';
            $data['error_string'][] = 'Password wajib diisi';
            $data['status'] = FALSE;
        }

        //   validasi repassword
        if ($this->input->post('repassword') == '') {
            $data['inputerror'][] = 'repassword';
            $data['error_string'][] = 'Harap ketik ulang password';
            $data['status'] = FALSE;
        } elseif (!$this->_validate_password($this->input->post('password'), $this->input->post('repassword'))) {
            $data['inputerror'][] = 'repassword';
            $data['error_string'][] = 'Password tidak sama, silakan periksa ulang';
            $data['status'] = FALSE;
        }

        if ($data['status'] === FALSE) {
            echo json_encode($data);
            exit();
        }
    }

    private function _validate_password($string1, $string2)
    {
        if ($string1 != $string2) {
            return FALSE;
        }

        return TRUE;
    }
}
