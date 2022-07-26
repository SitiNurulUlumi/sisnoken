<?php

class Auth extends CI_Controller
{

    public function __construct()
    {
        parent::__construct();
        $this->load->model('user_model');
        $this->load->helper(array('form', 'url'));
        $this->load->library(array('session', 'form_validation'));
    }

    public function index()
    {
        $this->login();
    }

    public function register()
    {
        $this->form_validation->set_rules('firstname', 'First Name', 'trim|required');
        $this->form_validation->set_rules('lastname', 'Last Name', 'trim|required');
        $this->form_validation->set_rules('email', 'Email', 'trim|required|is_unique[user.email]');
        $this->form_validation->set_rules('password', 'Password', 'trim|required|md5');
        $this->form_validation->set_rules('cpassword', 'Confirm Password', 'required|matches[password]|md5');


        $data['title'] = 'Register';

        if ($this->form_validation->run() === FALSE) {
            $this->load->view('templates/header', $data);
            $this->load->view('auth/register');
            $this->load->view('templates/footer');
        } else {
            if ($this->user_model->set_user()) {
                $this->session->set_flashdata('msg_success', 'Registration Successful!');
                redirect('auth/login');
            } else {
                $this->session->set_flashdata('msg_error', 'Error! Please try again later.');
                redirect('auth/register');
            }
        }
    }

    public function login()
    {
        if ($this->session->userdata('is_logged_in')) {
            redirect("home");
        }
        $email = $this->input->post('email');
        $password = $this->input->post('password');

        $this->form_validation->set_rules('email', 'Email', 'trim|required');
        $this->form_validation->set_rules('password', 'Password', 'trim|required|md5');

        $data['title'] = 'Login';

        if ($this->form_validation->run() === FALSE) {
            $this->load->view('templates/header', $data);
            $this->load->view('auth/login');
            $this->load->view('templates/footer');
        } else {
            if ($user = $this->user_model->get_user_login($email, $password)) {
                $this->session->set_userdata('email', $email);
                $this->session->set_userdata('user_id', $user['id']);
                $this->session->set_userdata('role', $user['role']);
                $this->session->set_userdata('is_logged_in', true);
                $this->session->set_userdata('is_info_read', false);

                $this->session->set_flashdata('msg_success', 'Login Successful!');

                switch ($this->session->userdata('role')) {
                    case '3':
                        redirect('queue/add');
                        break;
                    case '2':
                        redirect('home');
                        break;
                    case '1':
                        redirect('home');
                        break;

                    default:
                        redirect('home');
                        break;
                }
            } else {
                $this->session->set_flashdata('msg_error', 'Login credentials does not match!');

                $currentClass = $this->router->fetch_class(); // class = controller
                $currentAction = $this->router->fetch_method(); // action = function

                redirect("$currentClass/$currentAction");
                //redirect('user/login');
            }
        }
    }

    public function logout()
    {
        if ($this->session->userdata('is_logged_in')) {

            $this->session->unset_userdata('email');
            $this->session->unset_userdata('user_id');
            $this->session->unset_userdata('role');
            $this->session->unset_userdata('is_logged_in');
        }
        redirect('auth/');
    }
}
